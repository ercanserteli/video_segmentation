#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include "Superpixel.hpp"
#include <memory>


using namespace std;
using namespace cv;
using namespace cv::ximgproc;

void fillSuperpixelVector(Ptr<SuperpixelSLIC> slic, Mat& labels, vector<Superpixel>& superpixels, Mat& imageBGR, const int nBin1d);
void trainSVM(Ptr<ml::SVM>& svm, vector<Superpixel>& superpixels, ml::SVM::Types svmType, ml::SVM::KernelTypes kernelType);
Mat createFeatMat(vector<Superpixel*>& superpixels);
Mat createLabelsMat(const vector<Superpixel*>& superpixels);

class VideoSegmenter {
public:
	static const string trainingWindowName;
	static const string testWindowName;
	Mat trainLabels;
	float trainingAnnotationScale = 1.0f;

	Size getImageSize() const { return imTrain.size(); }
	bool setSuperpixelLabel(int i, int label, Vec3b overlayColor);
	void showImTrain() const;
	void showImTest() const;
	Mat getForeground(bool show) const;
	void cleanUp();
	void toggleTestOverlay() { showTestOverlay = !showTestOverlay; }
	VideoSegmenter(int superpixelSize, int superpixelRuler, int histoNbin1d, bool noiseReduction, bool spatialMomentum);
	~VideoSegmenter() {}

	void loadTrainInputsFromFile(Mat& imTrain, const std::string &inputPath);
	void loadPretrainedModel(const std::string &inputPath);
	void initialize(Mat& imTrain);
	void run(Mat& imTest);
	void showResults();
	void startTrainingAnnotation();

private:
	Ptr<SuperpixelSLIC> slicTrain;
	Ptr<SuperpixelSLIC> slicTest;
	Ptr<ml::SVM> SVMClassifier;
	vector<Superpixel> superpixelsTrain;
	vector<Superpixel> superpixelsTest;
	Mat imTrain;
	Mat imTrainOverlay;
	Mat imTest;
	Mat imTestOverlay;
	Mat prevForegroundMap;
	Mat prevBackgroundMap;
	bool showTestOverlay = true;
	int superpixelSize = 15;
	int superpixelRuler = 25;
	int histoNbin1d = 6;
	bool noiseReductionEnabled = false;
	bool spatialMomentumEnabled = false;
};