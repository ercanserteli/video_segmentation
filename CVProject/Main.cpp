#include "VideoSegmentation.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/ml.hpp>


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

		if (waitKey(1) >= 0) break;
	}
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
	//runVideo("C:/images/cat3_cr.mp4", "", "", "C:/images/abs.jpg", 20);
	//runVideo("C:/images/dancer.mkv", "dancer_svm2.xml", "", "");

	runWebcam("", 20);
	//runWebcam("C:/images/abs.jpg", 20);
	return 0;
}