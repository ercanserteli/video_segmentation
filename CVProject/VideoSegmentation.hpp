#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "Slic.hpp"
#include "Superpixel.hpp"
#include <memory>


using namespace std;
using namespace cv;

void fillSuperpixelVector(Slic& slic, vector<Superpixel>& superpixels, Mat& imageBGR, const int nBin1d = 6);
void trainSVM(Ptr<ml::SVM>& svm, vector<Superpixel>& superpixels, ml::SVM::Types svmType, ml::SVM::KernelTypes kernelType);
Mat createFeatMat(vector<Superpixel*>& superpixels);
Mat createLabelsMat(const vector<Superpixel*>& superpixels);

class VideoSegmenter {
public:
	struct Settings
	{
		int superpixelSize = 6;
		int superpixelCompact = 35;
		int histNbin1d = 6;
		int scaleBROI = 2;
		bool fullFrame = false;

		ml::SVM::KernelTypes kernelSVM = ml::SVM::RBF;
		ml::SVM::Types typeSVM = ml::SVM::C_SVC;
	};
	static const string trainingWindowName;
	static const string testWindowName;

	Mat getTrainLabels() const { return pSlicTrain->getLabels(); }
	Size getImageSize() const { return imTrain.size(); }
	bool setSuperpixelLabel(int i, int label, Vec3b overlayColor);
	void showImTrain() const;
	void showImTest() const;
	Mat showForeground() const;
	void cleanUp();
	void toggleTestOverlay() { showTestOverlay = !showTestOverlay; }
	VideoSegmenter(const Settings& settings);
	~VideoSegmenter() {}

	void loadTrainInputsFromFile(Mat& imTrain, const std::string &inputPath);
	void loadPretrainedModel(const std::string &inputPath);
	void initialize(Mat& imTrain);
	void run(Mat& imTest);
	void showResults();

private:
	unique_ptr<Slic> pSlicTrain;
	unique_ptr<Slic> pSlicTest;
	Ptr<ml::SVM> SVMClassifier;
	vector<Superpixel> superpixelsTrain;
	vector<Superpixel> superpixelsTest;
	Settings settings;
	Mat imTrain;
	Mat imTrainOverlay;
	Mat imTest;
	Mat imTestOverlay;
	Mat prevForegroundMap;
	bool showTestOverlay = true;
};