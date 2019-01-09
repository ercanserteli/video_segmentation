#pragma once

#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;


class Pixel
{
public:

	Point xy;
	Vec3f color;

	Pixel():xy(Point(-1, -1)), color(Vec3f(0, 0, 0)){}
	Pixel(Point xy, Vec3f color) :xy(xy), color(color){}
};
class Superpixel
{
public:

	Mat bgrHisto; // meanColor histogram
	Mat lbpHisto; // lbp histogram
	Vec3f meanColor;

	vector<Pixel> pixels;
	Mat image;
	Mat imageGray;
	Mat labels;
	Rect bounds;
	int id;
	uchar classLabel = 0;

	Superpixel(){}
	Superpixel(Vec3f color) :meanColor(color){}
	
	void computeFeatures(const int histoBin1d);
	void colorize(Mat& out, Vec3b color = Vec3b(255, 0, 0)) const;
	Mat getFeatMat();

private:
	void computeMeanAndBounds();
	void computeHisto(const int nBin1d);
	void computeLBPHisto();
};

