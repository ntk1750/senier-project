#include "stdafx.h"
#include "CamShift.h"
#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int getHistogram::go_histogram(cv::Mat applyImg, cv::Rect applyRoi)
{
	cv::Mat imageRoi = applyImg(applyRoi);

	Mat hsv, go_hue;
	
	int smin = 30, vmin = 10, vmax = 256;

	//std::cout << applyRoi.width << std::endl;

	if (imageRoi.data == NULL) {
		std::cout << "go_histgoram, imageRoi error";
		return -1;
	}

	//cvtColor 함수는 첫번째 인자값으로 넣어준 Mat형 변수를 3번째 인자값으로 바꿔 두번째 인자값으로 돌려주는 함수! 

//	cv::cvtColor(applyImg, hsv, cv::COLOR_BGR2HSV);

	cv::inRange(hsv, cv::Scalar(0, smin, MIN(vmin, vmax)),
		cv::Scalar(180, 256, MAX(vmin, vmax)), mask);

	int ch[] = { 0, 0 };
	hue.create(hsv.size(), hsv.depth());
	mixChannels(&hsv, 1, &hue, 1, ch, 1);
	/*IplImage *roi = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage *maskroi = cvCreateImage(size, IPL_DEPTH_8U, 3);*/

	cv::Mat roi(hue, applyRoi), maskroi(mask, applyRoi);
	/*roi = &IplImage(hue);
	maskroi = &IplImage(mask);

	cvSetImageROI(roi, rc);
	roiImage = cvarrToMat(roi);

	cvSetImageROI(maskroi, rc);
	roim = cvarrToMat(maskroi);
	*/
	cv::calcHist(&roi, 1, 0, maskroi, model, 1, &histSize, &ranges, true, false);
	cv::normalize(model, model, 0, 255, cv::NORM_MINMAX);

	trackRoi = applyRoi;
	
	return 0;
}

int getHistogram::go_tracking(cv::Mat img, cv::Rect& roi)
{
	//cv::cvtColor(img, hue, cv::COLOR_BGR2HSV);

	cv::calcBackProject(&hue, 1, 0, model, back, &ranges);

	back &= mask;

	cv::RotatedRect trackBox = cv::CamShift(back, trackRoi, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));

	//int cols = back.cols, rows = back.rows, r = (MIN(cols, rows) + 5) / 6;
	//trackRoi = cv::Rect(trackRoi.x - r, trackRoi.y - r, trackRoi.x + r, trackRoi.y + r) & cv::Rect(0, 0, cols, rows);

	cv::Mat back_frame;

	cv::cvtColor(back, back_frame, cv::COLOR_GRAY2BGR);
	cv::imshow("back", back_frame);


	cv::ellipse(img, trackBox, cv::Scalar(0, 255, 255), 3, cv::LINE_AA);

	cv::imshow("display", img);

	roi = trackRoi;

	return 0;
}