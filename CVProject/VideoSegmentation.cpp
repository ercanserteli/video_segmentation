#include "VideoSegmentation.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/ximgproc/slic.hpp>
#include "LambdaParallel.hpp"

const string VideoSegmenter::trainingWindowName = "Training selection";
const string VideoSegmenter::testWindowName = "Test results";

void padRect(Rect &rect, int padding, Size2i imgSize) {
	rect.x -= padding;
	rect.y -= padding;
	rect.width += padding;
	rect.height += padding;

	if (rect.x < 0) rect.x = 0;
	if (rect.y < 0) rect.y = 0;
	if (rect.width >= imgSize.width) rect.width = imgSize.width - 1;
	if (rect.height >= imgSize.height) rect.height = imgSize.height - 1;
}

// Toggles test overlay
void onImTestMouse(int event, int x, int y, int flags, void* data)
{
	if (event != EVENT_LBUTTONDOWN && event != EVENT_LBUTTONDBLCLK)
		return;

	VideoSegmenter* segmenter = static_cast<VideoSegmenter*>(data);
	segmenter->toggleTestOverlay();
	segmenter->showImTest();
}

// Training mouse annotation for training
void onImTrainMouse(int event, int x, int y, int flags, void* data)
{
	if (event != EVENT_LBUTTONDOWN && event != EVENT_RBUTTONDOWN && event != EVENT_MOUSEMOVE && event != EVENT_MBUTTONDOWN)
		return;

	VideoSegmenter* segmenter = static_cast<VideoSegmenter*>(data);
	const Size imSize = segmenter->getImageSize();
	const float scale = segmenter->trainingAnnotationScale;
	x = x / scale;
	y = y / scale;
	if (x < 0 || y < 0 || x >= imSize.width || y >= imSize.height)
		return;

	const int label = segmenter->trainLabels.at<int>(y, x);
	bool needRedraw = false;
	if (event == EVENT_MBUTTONDOWN) {
		needRedraw = segmenter->setSuperpixelLabel(label, 0, Vec3b(0, 0, 0));
	}
	else if (event == EVENT_LBUTTONDOWN) {
		needRedraw = segmenter->setSuperpixelLabel(label, 1, Vec3b(0, 255, 0));
	}
	else if (event == EVENT_RBUTTONDOWN) {
		needRedraw = segmenter->setSuperpixelLabel(label, 2, Vec3b(0, 0, 255));
	}
	else if (event == EVENT_MOUSEMOVE) {
		if ((flags & EVENT_FLAG_MBUTTON) > 0) {
			needRedraw = segmenter->setSuperpixelLabel(label, 0, Vec3b(0, 0, 0));
		}
		else if ((flags & EVENT_FLAG_LBUTTON) > 0) {
			needRedraw = segmenter->setSuperpixelLabel(label, 1, Vec3b(0, 255, 0));
		}
		else if ((flags & EVENT_FLAG_RBUTTON) > 0) {
			needRedraw = segmenter->setSuperpixelLabel(label, 2, Vec3b(0, 0, 255));
		}
	}

	if (needRedraw) segmenter->showImTrain();
}

// Creates Superpixel objects from existing SLIC superpixels, fills the given vector
void fillSuperpixelVector(Ptr<SuperpixelSLIC> slic, Mat& labels, vector<Superpixel>& superpixels, Mat& imageBGR, const int nBin1d)
{
	const int Nspx = slic->getNumberOfSuperpixels();
	superpixels.resize(Nspx);
	for (int i = 0; i < labels.rows; i++) {
		const Vec3b* image_ptr = imageBGR.ptr<Vec3b>(i);
		const int* label_ptr = labels.ptr<int>(i);
		for (int j = 0; j < labels.cols; j++) {
			superpixels[label_ptr[j]].pixels.push_back(Pixel(Point(j, i), Vec3f(image_ptr[j])));
		}
	}
	Mat imageGray;

	cvtColor(imageBGR, imageGray, CV_BGR2GRAY);

	parallel_for(Range(0, Nspx), [&](const Range& range) {
		for (int i = range.start; i < range.end; i++) {
			superpixels[i].image = imageBGR;
			superpixels[i].imageGray = imageGray;
			superpixels[i].id = i;
			superpixels[i].labels = labels;
			superpixels[i].computeFeatures(nBin1d);
		}
	});
}

// Creates an array (flat Mat) of labels from training superpixels
Mat createLabelsMat(const vector<Superpixel*>& superpixels)
{
	Mat labelsMat(superpixels.size(), 1, CV_32SC1, Scalar(0));
	int* labelsMat_ptr = (int*)labelsMat.data;
	for (int i = 0; i < superpixels.size(); i++) {
		if (superpixels[i]->classLabel != 0) {
			labelsMat_ptr[i] = superpixels[i]->classLabel;
		}
		else {
			cerr << "0 label is not allowed when training" << endl; 
		}
	}
	return labelsMat;
}

// Creates a feature matrix where rows are the superpixels and columns are their features
Mat createFeatMat(vector<Superpixel*>& superpixels)
{
	CV_Assert(!superpixels.empty());

	Mat featsMat;
	vector<Mat> featMats;
	featMats.reserve(superpixels.size());
	for (int i = 0; i < superpixels.size(); i++) {
		featMats.push_back(superpixels[i]->getFeatMat());
	}
	vconcat(featMats, featsMat);

	return featsMat;
}

void trainSVM(Ptr<ml::SVM>& svm, vector<Superpixel>& superpixels)
{
	CV_Assert(!superpixels.empty());
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	vector<Superpixel*> superpixel_ptrs;
	for (int i = 0; i < superpixels.size(); i++) {
		if (superpixels[i].classLabel != 0) {
			superpixel_ptrs.push_back(&superpixels[i]);
		}
	}
	const Mat fbSpxFeatMat = createFeatMat(superpixel_ptrs);
	const Mat labelsMat = createLabelsMat(superpixel_ptrs);
	const Ptr<ml::TrainData> tdata = ml::TrainData::create(fbSpxFeatMat, ml::ROW_SAMPLE, labelsMat);
	svm->trainAuto(tdata, 10);
}

bool VideoSegmenter::setSuperpixelLabel(int i, int label, Vec3b overlayColor)
{
	if (superpixelsTrain[i].classLabel == label) {
		return false;
	}
	superpixelsTrain[i].classLabel = label;

	superpixelsTrain[i].colorize(imTrainOverlay, overlayColor);
	return true;
}

void VideoSegmenter::showImTrain() const
{
	Mat result;
	Mat overlayMask;
	cvtColor(imTrainOverlay, overlayMask, CV_BGR2GRAY);
	overlayMask = ~(overlayMask > 0);

	addWeighted(imTrain, 0.5, imTrainOverlay, 0.5, 0, result);
	imTrain.copyTo(result, overlayMask);

	resize(result, result, Size(), trainingAnnotationScale, trainingAnnotationScale);
	imshow(trainingWindowName, result);
}

VideoSegmenter::VideoSegmenter(int superpixelSize, int superpixelCompactness, int histoNbin1d, bool noiseReduction, bool spatialMomentum)
{
	SVMClassifier = ml::SVM::create();
	this->superpixelSize = superpixelSize;
	this->superpixelRuler = superpixelCompactness;
	this->histoNbin1d = histoNbin1d;
	this->noiseReductionEnabled = noiseReduction;
	this->spatialMomentumEnabled = spatialMomentum;
}

void VideoSegmenter::startTrainingAnnotation() {
	int dim = MAX(imTrain.rows, imTrain.cols);
	if (dim < 500) {
		trainingAnnotationScale = 2;
	}
	else if(dim < 750){
		trainingAnnotationScale = 1.5;
	}
	else {
		trainingAnnotationScale = 1;
	}
	showImTrain();
	setMouseCallback(trainingWindowName, onImTrainMouse, static_cast<void*>(this));
}

void VideoSegmenter::initialize(Mat& imTrain)
{
	CV_Assert(imTrain.data != nullptr);

	this->imTrain = imTrain;
	//imshow("imTrain", imTrain);
	imTrainOverlay = Mat::zeros(this->imTrain.size(), CV_8UC3);

	//Superpixel segmentation
	slicTrain = createSuperpixelSLIC(this->imTrain, SLICO, superpixelSize, superpixelRuler);
	slicTrain->iterate(10);
	slicTrain->getLabels(trainLabels);

	//Create Superpixel vector from slic
	fillSuperpixelVector(slicTrain, trainLabels, superpixelsTrain, this->imTrain, histoNbin1d);

	//Painting contours
	Mat contours;
	slicTrain->getLabelContourMask(contours);
	this->imTrain.setTo(Vec3b(0, 0, 255), contours);

	startTrainingAnnotation();
	//Wait til annotating is over
	waitKey();
	//Annotating is over
	setMouseCallback(trainingWindowName, nullptr);

	if (spatialMomentumEnabled) {
		//Paint prevForegroundMap
		Mat foregroundMask = Mat::zeros(imTrain.size(), CV_8UC1);
		for (int i = 0; i < superpixelsTrain.size(); i++) {
			if (superpixelsTrain[i].classLabel == 1) {
				superpixelsTrain[i].colorize(foregroundMask, Vec3b(255, 255, 255));
			}
		}
		erode(~foregroundMask, prevBackgroundMap, getStructuringElement(CV_SHAPE_ELLIPSE, Size(7, 7)), Point(-1, -1), 2);
		erode(foregroundMask, prevForegroundMap, getStructuringElement(CV_SHAPE_ELLIPSE, Size(7, 7)), Point(-1, -1), 2);
	}

	int count1 = 0, count2 = 0;
	//Save inputs to file
	ofstream ofs("training_selections.txt");
	for (int i = 0; i < superpixelsTrain.size(); i++) {
		if (superpixelsTrain[i].classLabel == 2) {
			ofs << i << endl;
			++count1;
		}
	}
	ofs << "hede" << endl;
	for (int i = 0; i < superpixelsTrain.size(); i++) {
		if (superpixelsTrain[i].classLabel == 1) {
			ofs << i << endl;
			++count2;
		}
	}
	ofs.close();

	//There must be training samples for both classes
	CV_Assert(count1 > 0 && count2 > 0);

	//Train a classifier
	trainSVM(SVMClassifier, superpixelsTrain);
	SVMClassifier->save("svm.xml");
}

void VideoSegmenter::loadTrainInputsFromFile(Mat& imTrain, const std::string &inputPath)
{
	cout << "Loading training inputs from file: " << inputPath << endl;
	CV_Assert(imTrain.data != nullptr);

	this->imTrain = imTrain;
	imTrainOverlay = Mat::zeros(this->imTrain.size(), CV_8UC3);

	//Superpixel segmentation
	slicTrain = createSuperpixelSLIC(this->imTrain, SLICO, superpixelSize, superpixelRuler);
	slicTrain->iterate(10);
	slicTrain->getLabels(trainLabels);

	//Create Superpixel vector from slic
	fillSuperpixelVector(slicTrain, trainLabels, superpixelsTrain, this->imTrain, histoNbin1d);

	ifstream ifs(inputPath);
	string line;
	if (ifs.is_open()) {
		bool isFgnd = false;
		while (getline(ifs, line)) {
			if (line == "hede") {
				isFgnd = true;
			}
			else {
				if (isFgnd) {
					superpixelsTrain[atoi(line.c_str())].classLabel = 1;
				}
				else {
					superpixelsTrain[atoi(line.c_str())].classLabel = 2;
				}
			}
		}
		ifs.close();
	}
	if (spatialMomentumEnabled) {
		//Paint prevForegroundMap
		Mat foregroundMask = Mat::zeros(imTrain.size(), CV_8UC1);
		for (int i = 0; i < superpixelsTrain.size(); i++) {
			if (superpixelsTrain[i].classLabel == 1) {
				superpixelsTrain[i].colorize(foregroundMask, Vec3b(255, 255, 255));
			}
		}
		erode(~foregroundMask, prevBackgroundMap, getStructuringElement(CV_SHAPE_ELLIPSE, Size(7, 7)), Point(-1, -1), 2);
		erode(foregroundMask, prevForegroundMap, getStructuringElement(CV_SHAPE_ELLIPSE, Size(7, 7)), Point(-1, -1), 2);
	}

	trainSVM(SVMClassifier, superpixelsTrain);
}

void VideoSegmenter::loadPretrainedModel(const std::string &inputPath)
{
	cout << "Loading pre-trained model from file: " << inputPath << endl;
	SVMClassifier = Algorithm::load<ml::SVM>(inputPath);
}

void VideoSegmenter::run(Mat& imTest)
{
	CV_Assert(imTest.data != nullptr);

	if (spatialMomentumEnabled) {
		if (prevForegroundMap.empty()) {
			prevForegroundMap = Mat::zeros(imTest.size(), CV_32FC1);
		}
		if (prevBackgroundMap.empty()) {
			prevBackgroundMap = Mat::zeros(imTest.size(), CV_32FC1);
		}
	}

	this->imTest = imTest;
	imTestOverlay = Mat::zeros(this->imTest.size(), CV_8UC3);

	//Superpixel segmentation on the test image
	slicTest = createSuperpixelSLIC(this->imTest, MSLIC, superpixelSize, superpixelRuler);
	slicTest->iterate(10);
	Mat testLabels;
	slicTest->getLabels(testLabels);

	//Create Superpixel vector from slic
	fillSuperpixelVector(slicTest, testLabels, superpixelsTest, this->imTest, histoNbin1d);
	
	Mat foregroundMask = Mat::zeros(imTest.size(), CV_8UC1);
	for (int i = 0; i < superpixelsTest.size(); i++) {
		//float response = SVMClassifier->predict(superpixelsTest[i].getFeatMat());
		float responseRaw = SVMClassifier->predict(superpixelsTest[i].getFeatMat(), noArray(), ml::StatModel::RAW_OUTPUT);
		if (spatialMomentumEnabled) {
			responseRaw += prevForegroundMap.at<uchar>(superpixelsTest[i].centroid)*0.25 - prevBackgroundMap.at<uchar>(superpixelsTest[i].centroid)*0.25;
		}
		uchar label = (responseRaw > 0 ? 1 : 2);
		superpixelsTest[i].classLabel = label;
		if (superpixelsTest[i].classLabel == 1) {
			superpixelsTest[i].colorize(foregroundMask, Vec3b(255, 255, 255));
		}
	}

	if (noiseReductionEnabled) {
		Mat kernel = getStructuringElement(CV_SHAPE_ELLIPSE, Size(3, 3));
		for (int i = 0; i < superpixelsTest.size(); i++) {
			Rect bounds = superpixelsTest[i].bounds;
			padRect(bounds, 1, imTest.size());
			Mat mask = (testLabels(bounds) == i);
			Mat maskBorders;
			dilate(mask, maskBorders, kernel);
			addWeighted(maskBorders, 1, mask, -1, 0, maskBorders);
			if (superpixelsTest[i].classLabel == 1) {
				int count = countNonZero(foregroundMask(bounds) & maskBorders);
				if (count <= 2) {
					superpixelsTest[i].classLabel = 2;
					superpixelsTest[i].colorize(foregroundMask, Vec3b(0, 0, 0));
				}
			}
			else {
				int count = countNonZero(~(foregroundMask(bounds)) & maskBorders);
				if (count <= 1) {
					superpixelsTest[i].classLabel = 1;
					superpixelsTest[i].colorize(foregroundMask, Vec3b(255, 255, 255));
				}
			}
		}
	}
	if (spatialMomentumEnabled) {
		//dilate(prevForegroundMask, prevForegroundMask, getStructuringElement(CV_SHAPE_ELLIPSE, Size(7, 7)), Point(-1, -1), 3);
		//distanceTransform(prevForegroundMask, prevForegroundMap, DIST_L2, 3, CV_32F);
		erode(~foregroundMask, prevBackgroundMap, getStructuringElement(CV_SHAPE_ELLIPSE, Size(7, 7)), Point(-1, -1), 2);
		erode(foregroundMask, prevForegroundMap, getStructuringElement(CV_SHAPE_ELLIPSE, Size(7, 7)), Point(-1, -1), 2);
		//normalize(prevForegroundMap, prevForegroundMap, -0.5, 0.5, NORM_MINMAX);
	}
}

void VideoSegmenter::showResults()
{
	//Classification result
	for (int i = 0; i < superpixelsTest.size(); i++) {
		if (superpixelsTest[i].classLabel == 1)superpixelsTest[i].colorize(imTestOverlay, Vec3b(0, 255, 0));
		else if (superpixelsTest[i].classLabel == 2)superpixelsTest[i].colorize(imTestOverlay, Vec3b(0, 0, 255));
		else if (superpixelsTest[i].classLabel == 3)superpixelsTest[i].colorize(imTestOverlay, Vec3b(255, 0, 0));
	}

	//Painting contours
	Mat contours;
	slicTest->getLabelContourMask(contours);
	imTestOverlay.setTo(Vec3b(0, 0, 255), contours);
	showImTest();
	setMouseCallback(testWindowName, onImTestMouse, static_cast<void*>(this));

	waitKey();
}

void VideoSegmenter::showImTest() const
{
	if (showTestOverlay) {
		Mat result;
		Mat overlayMask;
		cvtColor(imTestOverlay, overlayMask, CV_BGR2GRAY);
		overlayMask = ~(overlayMask > 0);

		addWeighted(imTest, 0.5, imTestOverlay, 0.5, 0, result);
		imTest.copyTo(result, overlayMask);
		imshow(testWindowName, result);
	}
	else {
		imshow(testWindowName, imTest);
	}
}

Mat VideoSegmenter::getForeground(bool show) const
{
	Mat foreground = imTest.clone();
	for (int i = 0; i < superpixelsTest.size(); i++) {
		if (superpixelsTest[i].classLabel != 1)superpixelsTest[i].colorize(foreground, Vec3b(0, 0, 0));
	}
	if (show) {
		imshow(testWindowName, foreground);
	}
	return foreground;
}

void VideoSegmenter::cleanUp()
{
	superpixelsTest.clear();
}