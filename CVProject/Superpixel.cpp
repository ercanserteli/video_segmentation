#include "Superpixel.hpp"

using namespace std;
using namespace cv;

void Superpixel::computeFeatures(const int histoBin1d) {
	computeMeanAndBounds();
	computeHisto(histoBin1d);
	computeLBPHisto();
}


void Superpixel::computeMeanAndBounds() {
	if (pixels.size() != 0) {
		int minX = INT_MAX, minY = INT_MAX, maxX = 0, maxY = 0;
		for (const Pixel px : pixels) {
			if (px.pos.x < minX) minX = px.pos.x;
			if (px.pos.x > maxX) maxX = px.pos.x;
			if (px.pos.y < minY) minY = px.pos.y;
			if (px.pos.y > maxY) maxY = px.pos.y;
			meanColor += px.color;
			centroid += px.pos;
		}
		meanColor /= static_cast<float>(pixels.size());
		centroid /= static_cast<float>(pixels.size());
		bounds = Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
	}
}

void Superpixel::computeHisto(const int nBin1d)
{
	Mat pxMat(pixels.size(), 1, CV_32FC3);
	Vec3f* pxMat_ptr = pxMat.ptr<Vec3f>();
	for (int i = 0; i < pixels.size(); i++) {
		pxMat_ptr[i] = pixels[i].color;
	}
	CV_Assert(pxMat.isContinuous());

	int histSize[] = { nBin1d, nBin1d, nBin1d };

	float range[2] = { 0, 256 };
	const float* ranges[] = { range, range, range };

	int channels[] = { 0, 1, 2 };

	calcHist(&pxMat, 1, channels, Mat(), bgrHisto, 3, histSize, ranges, true, false);

	//normalization 3D
	float* hist_ptr = bgrHisto.ptr<float>();
	int a, b;
	int col_row = bgrHisto.size[0] * bgrHisto.size[1];
	int facNorm = nBin1d * nBin1d * nBin1d;
	for (int k = 0; k < bgrHisto.size[2]; k++)
	{
		a = k * col_row;
		for (int i = 0; i < bgrHisto.size[0]; i++)
		{
			b = i * bgrHisto.size[1];
			for (int j = 0; j < bgrHisto.size[1]; j++)
			{
				hist_ptr[a + b + j] /= facNorm;
			}
		}
	}
}

void Superpixel::computeLBPHisto()
{
	lbpHisto = Mat::zeros(256, 1, CV_32FC1);
	for (int j = 0; j < pixels.size(); j++)
	{
		const int x = pixels[j].pos.x;
		const int y = pixels[j].pos.y;

		if (x > 0 && y > 0 && x < imageGray.cols - 1 && y < imageGray.rows - 1) {
			const uchar val = imageGray.at<uchar>(y, x);
			uchar histoValue = 0;

			if (imageGray.at<uchar>(y, x - 1) >= val) histoValue += 128;

			if (imageGray.at<uchar>(y - 1, x - 1) >= val) histoValue += 64;

			if (imageGray.at<uchar>(y - 1, x) >= val) histoValue += 32;

			if (imageGray.at<uchar>(y - 1, x + 1) >= val) histoValue += 16;

			if (imageGray.at<uchar>(y, x + 1) >= val) histoValue += 8;

			if (imageGray.at<uchar>(y + 1, x + 1) >= val) histoValue += 4;

			if (imageGray.at<uchar>(y + 1, x) >= val) histoValue += 2;

			if (imageGray.at<uchar>(y + 1, x - 1) >= val) histoValue += 1;

			lbpHisto.at<float>(histoValue) += 1.f;
		}
	}
}

void Superpixel::colorize(Mat& out, Vec3b color) const{
	if (out.channels() == 3){
		out.setTo(color, labels == id);
	}
	else{
		out.setTo(color[0], labels == id);
	}
}

Mat Superpixel::getFeatMat()
{
	Mat feats[3];
	Mat feat_total;

	feats[0] = (Mat_<float>(1, 3) << meanColor[0], meanColor[1], meanColor[2]);
	normalize(feats[0], feats[0], 1, 0);

	feats[1] = Mat(1, bgrHisto.size[0] * bgrHisto.size[1] * bgrHisto.size[2], CV_32F, bgrHisto.ptr<float>());
	normalize(feats[1], feats[1], 1, 0);

	feats[2] = Mat(1, lbpHisto.size[0], CV_32F, lbpHisto.ptr<float>());
	normalize(feats[2], feats[2], 1, 0);

	hconcat(feats, 3, feat_total);
	return feat_total;
}

