#include "VideoSegmentation.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>

namespace fs = std::filesystem;


void runSingle(Mat& imageTrain, Mat& imageTest, const string& modelFilePath = "", const string& trainingRoisFilePath = "")
{
	//imshow("imageTrain", imageTrain);
	VideoSegmenter segmenter(15, 15, 6, false, false);
	if (trainingRoisFilePath.empty()) {
		if (modelFilePath.empty()) {
			segmenter.initialize(imageTrain); // train
		}
		else {
			segmenter.loadPretrainedModel(modelFilePath);
		}
	}
	else {
		segmenter.loadTrainInputsFromFile(imageTrain, trainingRoisFilePath);
	}

	segmenter.run(imageTrain); // test
	segmenter.showResults();
	segmenter.cleanUp();
	segmenter.run(imageTest); // test
	segmenter.showResults();
}

void runWebcam(const string& bgFilePath = "", int superpixelSize = 10)
{
	VideoCapture vc(0);

	if (!vc.isOpened()) {  // check if we succeeded
		cerr << "VideoCapture could not be opened" << endl;
		return;
	}

	VideoSegmenter segmenter(superpixelSize, 15, 6, false, false);

	Mat frame;
	int key = 0;
	do {
		vc >> frame;
		imshow("Camera", frame);
		key = waitKey(33);
	} while (key == -1);

	Mat bg;
	if (!bgFilePath.empty()) {
		bg = imread(bgFilePath);
		Size frameSize = frame.size();
		Size bgSize = bg.size();
		CV_Assert(bgSize.height >= frameSize.height && bgSize.width >= frameSize.width);
		bg = bg(Rect((bgSize.width - frameSize.width) / 2, (bgSize.height - frameSize.height) / 2, frameSize.width, frameSize.height));
	}

	segmenter.initialize(frame); // train

	for (;;)
	{
		vc >> frame;

		if (frame.empty()) {
			return;
		}

		segmenter.run(frame); // test

		Mat fg = segmenter.getForeground(false);
		if (!bg.empty()) {
			Mat newFrame = bg.clone();
			Mat fgMask;
			cvtColor(fg, fgMask, COLOR_BGR2GRAY);

			fg.copyTo(newFrame, fgMask);
			fg = newFrame;
		}
		imshow("Camera", frame);
		imshow("Output", fg);
		segmenter.cleanUp();

		if (waitKey(1) >= 0) break;
	}
}

void runVideo(const string& videoFilePath, const string& modelFilePath = "", const string& trainingRoisFilePath = "", const string& bgFilePath = "", int superpixelSize = 10)
{
	VideoCapture vc(videoFilePath);

	if (!vc.isOpened()) {  // check if we succeeded
		cerr << "VideoCapture could not be opened" << endl;
		return;
	}
	VideoWriter vw("segmented.mp4", vc.get(CV_CAP_PROP_FOURCC), vc.get(CV_CAP_PROP_FPS), Size(vc.get(CV_CAP_PROP_FRAME_WIDTH), vc.get(CV_CAP_PROP_FRAME_HEIGHT)));
	VideoSegmenter segmenter(superpixelSize, 15, 6, true, true);

	Mat frame;
	vc >> frame;
	//imshow("Frame", frame);

	Mat bg;
	if (!bgFilePath.empty()) {
		bg = imread(bgFilePath);
		Size frameSize = frame.size();
		Size bgSize = bg.size();
		CV_Assert(bgSize.height >= frameSize.height && bgSize.width >= frameSize.width);
		bg = bg(Rect((bgSize.width - frameSize.width)/2, (bgSize.height - frameSize.height) / 2, frameSize.width, frameSize.height));
	}
	
	if (trainingRoisFilePath.empty()) {
		if (modelFilePath.empty()) {
			segmenter.initialize(frame); // train
		}
		else {
			segmenter.loadPretrainedModel(modelFilePath);
		}
	}
	else {
		segmenter.loadTrainInputsFromFile(frame, trainingRoisFilePath);
	}

	for (;;)
	{
		vc >> frame;

		if (frame.empty()) {
			return;
		}

		segmenter.run(frame); // test

		Mat fg = segmenter.getForeground(true);
		if (!bg.empty()) {
			Mat newFrame = bg.clone();
			Mat fgMask;
			cvtColor(fg, fgMask, COLOR_BGR2GRAY);

			fg.copyTo(newFrame, fgMask);
			fg = newFrame;
		}
		//imshow("Output", fg);
		vw << fg;
		segmenter.cleanUp();
	}
}

// Tests the algorithm on a sample in the DAVIS data set
void testDavis(const string& name , const string& modelFilePath = "", int superpixelSize = 10, bool noiseReduction = true, bool temporalMomentum = false) {
	char imageSequencePattern[260];
	char outputSequencePattern[260];
	char outputFolder[260];
	char annotationPath[260];

	sprintf_s(imageSequencePattern, "C:\\images\\DAVIS\\JPEGImages\\480p\\%s\\%%05d.jpg", name.c_str());
	sprintf_s(outputSequencePattern, "C:\\images\\DAVIS\\output\\%s\\%%05d.png", name.c_str());
	sprintf_s(annotationPath, "C:\\images\\DAVIS\\Annotations\\480p\\%s\\00000.png", name.c_str());
	sprintf_s(outputFolder, "C:\\images\\DAVIS\\output\\%s", name.c_str());
	
	// Create output folder if it does not exist
	if (!fs::exists(outputFolder)) {
		fs::create_directory(outputFolder);
	}

	VideoCapture vc(imageSequencePattern);

	if (!vc.isOpened())  // check if we succeeded
		return;

	VideoSegmenter segmenter(superpixelSize, 15, 6, noiseReduction, temporalMomentum);

	Mat frame;
	vc >> frame;
	Mat annotationMask = imread(annotationPath, IMREAD_GRAYSCALE);

	if (modelFilePath.empty()) {
		if (annotationMask.empty()) {
			segmenter.initialize(frame); // train manually
		}
		else {
			segmenter.initialize(frame, annotationMask); // auto train using the first annotation frame
		}
	}
	else {
		segmenter.loadPretrainedModel(modelFilePath);
	}

	int frameId = 0;
	for (;;)
	{

		if (frame.empty()) {
			return;
		}

		segmenter.run(frame); // test

		Mat fgmask = segmenter.getForegroundMask();
		//imshow("Output", fg);
		char buf[260];
		sprintf_s(buf, outputSequencePattern, frameId);
		imwrite(buf, fgmask);
		segmenter.cleanUp();

		if (waitKey(1) >= 0) break;

		vc >> frame;
		frameId++;
	}
}

// Calculates and prints Dice and Jaccard scores on results of a DAVIS test
float calculateScores(const string& name) {
	char truthSequencePattern[260];
	char predictedSequencePattern[260];

	sprintf_s(truthSequencePattern, "C:\\images\\DAVIS\\Annotations\\480p\\%s\\%%05d.png", name.c_str());
	sprintf_s(predictedSequencePattern, "C:\\images\\DAVIS\\output\\%s\\%%05d.png", name.c_str());

	VideoCapture truth(truthSequencePattern);
	VideoCapture pred(predictedSequencePattern);

	if (!truth.isOpened() || !pred.isOpened())  // check if we succeeded
		return -1;

	Mat truthFrame, predFrame;
	float totalDice = 0, totalJaccard = 0;
	int frameCount = 0;
	for (;;)
	{
		truth >> truthFrame;
		pred >> predFrame;

		if (truthFrame.channels() == 4) {
			cvtColor(truthFrame, truthFrame, COLOR_BGRA2GRAY);
		}
		else if (truthFrame.channels() == 3) {
			cvtColor(truthFrame, truthFrame, COLOR_BGR2GRAY);
		}

		if (truthFrame.empty() || predFrame.empty()) {
			break;
		}
		frameCount++;

		Mat intersectionMat, unionMat;
		bitwise_and(truthFrame, predFrame, intersectionMat);
		bitwise_or(truthFrame, predFrame, unionMat);
		int truthCount = countNonZero(truthFrame);
		int predCount = countNonZero(predFrame);
		int interCount = countNonZero(intersectionMat);
		int unionCount = countNonZero(unionMat);

		float dice = (2.f * interCount) / static_cast<float>(truthCount + predCount);
		float jaccard = static_cast<float>(interCount) / static_cast<float>(unionCount);
		totalDice += dice;
		totalJaccard += jaccard;
	}

	float meanDice = totalDice / frameCount;
	float meanJaccard = totalJaccard / frameCount;
	cout << "Dice score for " << name << ": " << meanDice << endl;
	cout << "Jaccard score for " << name << ": " << meanJaccard << endl;
}

// Runs tests on every sample in the DAVIS dataset
void runDavisTests() {
	for (const auto & entry : fs::directory_iterator("C:\\images\\DAVIS\\JPEGImages\\480p")) {
		string name = entry.path().filename().generic_string();
		cout << "Testing on " << entry.path().filename().generic_string() << endl;
		testDavis(name);
		calculateScores(name);
	}
	getchar();
}


int main(int argc, char** argv)
{
	/*Mat train = imread("frame0.png");
	Mat test = imread("frame1.png");
	runSingle(train, test, "", "training_selections.txt");*/
	//runVideo("C:/images/me_cr.mp4");
	//runVideo("C:/images/me_cr.mp4", "dance_cr_new_svm.xml", "", "C:/images/sponge.jpg");
	//runVideo("C:/images/kettleman.mp4", "kettleman_svm.xml", "", "C:/images/sponge.jpg");
	//runVideo("C:/images/catn.mp4", "catn_svm.xml", "", "C:/images/abs.jpg");
	//runVideo("C:/images/dancer.mkv", "dancer_svm2.xml", "", "");

	//runWebcam("C:/images/abs.jpg", 20);
	runDavisTests();
	return 0;
}