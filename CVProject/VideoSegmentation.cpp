#include "VideoSegmentation.hpp"
#include <iostream>
#include <fstream>
//#include "Slic.hpp"
#include "LambdaParallel.hpp"
#include <opencv2/ximgproc/slic.hpp>

const string VideoSegmenter::trainingWindowName = "Training selection";
const string VideoSegmenter::testWindowName = "Test results";

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
	if (x < 0 || y < 0 || x >= imSize.width || y >= imSize.height)
		return;

	Mat trainLabels = segmenter->getTrainLabels();
	const int label = trainLabels.at<int>(y, x);
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
void fillSuperpixelVector(Slic& slic, vector<Superpixel>& superpixels, Mat& imageBGR, const int nBin1d)
{
	const int Nspx = slic.getNbSpx();
	superpixels.resize(Nspx);
	const Mat labels = slic.getLabels();
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
			superpixels[i].labels = slic.getLabels();
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

void trainSVM(Ptr<ml::SVM>& svm, vector<Superpixel>& superpixels, ml::SVM::Types svmType, ml::SVM::KernelTypes kernelType)
{
	CV_Assert(!superpixels.empty());
	svm->setType(svmType);
	svm->setKernel(kernelType);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	vector<Superpixel*> superpixels_ptr;
	for (int i = 0; i < superpixels.size(); i++) {
		if (superpixels[i].classLabel != 0) {
			superpixels_ptr.push_back(&superpixels[i]);
		}
	}
	const Mat fbSpxFeatMat = createFeatMat(superpixels_ptr);
	const Mat labelsMat = createLabelsMat(superpixels_ptr);
	const Ptr<ml::TrainData> tdata = ml::TrainData::create(fbSpxFeatMat, ml::ROW_SAMPLE, labelsMat);
	svm->trainAuto(tdata);
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
	imshow(trainingWindowName, result);
}

VideoSegmenter::VideoSegmenter(const Settings& settings)
{
	pSlicTrain = make_unique<Slic>();
	pSlicTest = make_unique<Slic>();
	SVMClassifier = ml::SVM::create();
	this->settings = settings;
}

void VideoSegmenter::initialize(Mat& imTrain)
{
	CV_Assert(imTrain.data != nullptr);

	this->imTrain = imTrain;
	prevForegroundMap = Mat::zeros(imTrain.size(), CV_32FC1);
	//imshow("imTrain", imTrain);
	imTrainOverlay = Mat::zeros(this->imTrain.size(), CV_8UC3);

	//Superpixel segmentation
	pSlicTrain->initialize(this->imTrain, settings.superpixelSize, settings.superpixelCompact, 10, Slic::SLIC_SIZE);
	pSlicTrain->generateSpx(this->imTrain);

	//Create Superpixel vector from slic
	fillSuperpixelVector(*pSlicTrain, superpixelsTrain, this->imTrain);

	pSlicTrain->display_contours(this->imTrain);

	imshow(trainingWindowName, this->imTrain);
	setMouseCallback(trainingWindowName, onImTrainMouse, static_cast<void*>(this));
	//Wait til annotating is over
	waitKey();
	//Annotating is over
	setMouseCallback(trainingWindowName, nullptr);

	//Paint prevForegroundMap
	Mat prevForegroundMask = Mat::zeros(imTest.size(), CV_8UC1);
	for (int i = 0; i < superpixelsTrain.size(); i++) {
		if (superpixelsTrain[i].classLabel == 1) {
			superpixelsTrain[i].colorize(prevForegroundMask, Vec3b(255, 255, 255));
		}
	}
	dilate(prevForegroundMask, prevForegroundMask, getStructuringElement(CV_SHAPE_ELLIPSE, Size(7, 7)));
	distanceTransform(prevForegroundMask, prevForegroundMap, DIST_L2, 3, CV_32F);

	//Save inputs to file
	ofstream ofs("training_selections.txt");
	for (int i = 0; i < superpixelsTrain.size(); i++) {
		if (superpixelsTrain[i].classLabel == 2) {
			ofs << i << endl;
		}
	}
	ofs << "hede" << endl;
	for (int i = 0; i < superpixelsTrain.size(); i++) {
		if (superpixelsTrain[i].classLabel == 1) {
			ofs << i << endl;
		}
	}
	ofs.close();

	//Train a classifier
	trainSVM(SVMClassifier, superpixelsTrain, settings.typeSVM, settings.kernelSVM);
	SVMClassifier->save("svm.xml");
}

void VideoSegmenter::loadTrainInputsFromFile(Mat& imTrain, const std::string &inputPath)
{
	cout << "Loading training inputs from file: " << inputPath << endl;
	CV_Assert(imTrain.data != nullptr);

	this->imTrain = imTrain;
	prevForegroundMap = Mat::zeros(imTrain.size(), CV_32FC1);
	imTrainOverlay = Mat::zeros(this->imTrain.size(), CV_8UC3);

	//Superpixel segmentation
	pSlicTrain->initialize(this->imTrain, settings.superpixelSize, settings.superpixelCompact, 10, Slic::SLIC_SIZE);
	pSlicTrain->generateSpx(this->imTrain);

	//Create Superpixel vector from slic
	fillSuperpixelVector(*pSlicTrain, superpixelsTrain, this->imTrain);

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

	//Paint prevForegroundMap
	Mat prevForegroundMask = Mat::zeros(imTest.size(), CV_8UC1);
	for (int i = 0; i < superpixelsTrain.size(); i++) {
		if(superpixelsTrain[i].classLabel == 1){
			superpixelsTrain[i].colorize(prevForegroundMask, Vec3b(255, 255, 255));
		}
	}
	dilate(prevForegroundMask, prevForegroundMask, getStructuringElement(CV_SHAPE_ELLIPSE, Size(7, 7)));
	distanceTransform(prevForegroundMask, prevForegroundMap, DIST_L2, 3, CV_32F);

	trainSVM(SVMClassifier, superpixelsTrain, settings.typeSVM, settings.kernelSVM);
}

void VideoSegmenter::loadPretrainedModel(const std::string &inputPath)
{
	cout << "Loading pre-trained model from file: " << inputPath << endl;
	SVMClassifier = Algorithm::load<ml::SVM>(inputPath);
}

void VideoSegmenter::run(Mat& imTest)
{
	CV_Assert(imTest.data != nullptr);

	if (prevForegroundMap.empty()) {
		prevForegroundMap = Mat::zeros(imTest.size(), CV_32FC1);
	}

	this->imTest = imTest;
	imTestOverlay = Mat::zeros(this->imTest.size(), CV_8UC3);

	pSlicTest->initialize(this->imTest, settings.superpixelSize, settings.superpixelCompact, 10, Slic::SLIC_SIZE);
	pSlicTest->generateSpx(this->imTest);

	fillSuperpixelVector(*pSlicTest, superpixelsTest, this->imTest);
	imshow("prevForegroundMap", prevForegroundMap);

	Mat prevForegroundMask = Mat::zeros(imTest.size(), CV_8UC1);
	for (int i = 0; i < superpixelsTest.size(); i++) {
		float response = SVMClassifier->predict(superpixelsTest[i].getFeatMat());
		superpixelsTest[i].classLabel = static_cast<uchar>(static_cast<int>(response));
		if(superpixelsTest[i].classLabel == 1){
			superpixelsTest[i].colorize(prevForegroundMask, Vec3b(255, 255, 255));
		}
	}

	dilate(prevForegroundMask, prevForegroundMask, getStructuringElement(CV_SHAPE_ELLIPSE, Size(7, 7)));
	distanceTransform(prevForegroundMask, prevForegroundMap, DIST_L2, 3, CV_32F);
}

void VideoSegmenter::showResults()
{
	//Classification result
	for (int i = 0; i < superpixelsTest.size(); i++) {
		if (superpixelsTest[i].classLabel == 1)superpixelsTest[i].colorize(imTestOverlay, Vec3b(0, 255, 0));
		else if (superpixelsTest[i].classLabel == 2)superpixelsTest[i].colorize(imTestOverlay, Vec3b(0, 0, 255));
		else if (superpixelsTest[i].classLabel == 3)superpixelsTest[i].colorize(imTestOverlay, Vec3b(255, 0, 0));
	}
	pSlicTest->display_contours(imTestOverlay);
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

Mat VideoSegmenter::showForeground() const
{
	Mat foreground = imTest.clone();
	for (int i = 0; i < superpixelsTest.size(); i++) {
		if (superpixelsTest[i].classLabel != 1)superpixelsTest[i].colorize(foreground, Vec3b(0, 0, 0));
	}
	imshow(testWindowName, foreground);
	return foreground;
}

void VideoSegmenter::cleanUp()
{
	superpixelsTest.clear();
}