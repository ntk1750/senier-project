#pragma once
#include <opencv2\opencv.hpp>

class getHistogram
{
private:
	int bin_w;
	int histSize = 16;
	float hsvRange[2] = { 0, 180 };
	const float* ranges = { hsvRange };
	int channl[2] = { 0, 1 };

	cv::Rect trackRoi;
	cv::Mat model, back, hue;
	cv::Mat mask;

public:
	int go_histogram(cv::Mat applyImg, cv::Rect applyRoi);
	int go_tracking(cv::Mat img, cv::Rect& roi);
};