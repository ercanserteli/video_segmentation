#include "VideoSegmentation.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/ml.hpp>


void runSingle(Mat& imageTrain, Mat& imageTest, const string& modelFilePath = "", const string& trainingRoisFilePath = "")
{
	VideoSegmenter::Settings settings;
	settings.superpixelSize = 15;
	settings.superpixelCompact = 25; // lower values results in irregular shapes
	settings.histNbin1d = 16;
	settings.scaleBROI = 2;
	settings.fullFrame = false;
	settings.kernelSVM = ml::SVM::RBF;
	settings.typeSVM = ml::SVM::C_SVC;
	//imshow("imageTrain", imageTrain);
	VideoSegmenter segmenter(settings);
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

void runVideo(const string& videoFilePath, const string& modelFilePath = "", const string& trainingRoisFilePath = "")
{
	VideoCapture vc(videoFilePath);

	if (!vc.isOpened())  // check if we succeeded
		return;

	VideoWriter vw("segmented.mp4", vc.get(CV_CAP_PROP_FOURCC), vc.get(CV_CAP_PROP_FPS), Size(vc.get(CV_CAP_PROP_FRAME_WIDTH), vc.get(CV_CAP_PROP_FRAME_HEIGHT)));

	VideoSegmenter::Settings settings;
	//settings.superpixelSize = 15;
	settings.superpixelSize = 10;
	settings.superpixelCompact = 25; // lower values results in irregular shapes
	settings.histNbin1d = 16;
	settings.scaleBROI = 2;
	settings.fullFrame = false;
	settings.kernelSVM = ml::SVM::RBF;
	settings.typeSVM = ml::SVM::C_SVC;

	VideoSegmenter segmenter(settings);

	Mat frame;
	vc >> frame; // get a new frame from camera

	
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
		vc >> frame; // get a new frame from camera

		if (frame.empty()) {
			return;
		}

		segmenter.run(frame); // test

		Mat fg = segmenter.showForeground();
		vw << fg;
		segmenter.cleanUp();

		if (waitKey(1) >= 0) break;
	}
}


int main(int argc, char** argv)
{
	/*Mat train = imread("frame0.png");
	Mat test = imread("frame1.png");
	runSingle(train, test, "", "training_selections.txt");*/
	//runVideo("C:/images/dance.mp4", "", "dance_selections.txt");
	runVideo("C:/images/dance_cr.mp4", "dance_cr_svm.xml", "");
	return 0;
}