#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <opencv\highgui.h>
#include <iostream>
#include <time.h>
#include "CamShift.h"
#include "BlobLabeling.h"

#include<Windows.h>

#include<stdint.h>
using namespace cv;
using namespace std;
#pragma comment(lib,"winmm.lib")



#define SOUND_FILE_NAME "C:\Users\DST\Desktop\CamShift\CamShift\CamShift\sonarbeep.wav"
struct CallMouseInfo
{
	cv::Mat m_frame;//이미지 파일 저장
	cv::Rect m_roi;//지정영역
	cv::Point m_tmp, m_tmp1;//해당 영역지정

	bool onDrag = false;
	bool hist_ready = false;
};
void onMouse(int event, int x, int y, int flags, void* data)
{
	CallMouseInfo *callMouse = (CallMouseInfo *)data;

	/*if (callMouse->onDrag) {

		callMouse->m_roi.x = MIN(x, callMouse->m_tmp.x);
		callMouse->m_roi.y = MIN(y, callMouse->m_tmp.y);

		callMouse->m_roi.width = std::abs(x - callMouse->m_tmp.x);
		callMouse->m_roi.height = std::abs(y - callMouse->m_tmp.y);

		callMouse->m_roi &= cv::Rect(0, 0, callMouse->m_frame.cols, callMouse->m_frame.rows);


		cv::Mat image;
		image = callMouse->m_frame;

		callMouse->m_tmp1.x = x;
		callMouse->m_tmp1.y = y;

		cv::rectangle(image, callMouse->m_tmp, callMouse->m_tmp1, cv::Scalar(0, 255, 0), 1);

		cv::imshow("display", image);
	}

	if (event == CV_EVENT_LBUTTONDOWN) {

		// Ã¹ ½ÃÀÛÀÌ µÇ´Â X, Y ÁÂÇ¥
		callMouse->m_tmp.x = x;
		callMouse->m_tmp.y = y;

		callMouse->m_roi = cv::Rect(x, y, 0, 0);

		callMouse->onDrag = true;
	}

	if (event == CV_EVENT_LBUTTONUP) {
		callMouse->onDrag = false;
		callMouse->hist_ready = true;

	}
	*/


}
void InitFrameBuffer(Mat* buffer[], CvSize size)
{
	for (int i = 0; i<3; i++) {
		//buffer[i] = cvCreateImage(size, IPL_DEPTH_8U, 1);
		buffer[i]->create(size, CV_8UC1);
		cvZero(buffer[i]);
	}
}
void ReleaseFrameBuffer(Mat *buffer[])
{
	for (int i = 0; i < 3; i++){
		//cvReleaseImage(&buffer[i]);
		buffer[i]->release();
	}
}
Mat objectHistogram;
Mat globalHistogram;
void getObjectHistogram(Mat &frame, Rect object_region)
{
	const int channels[] = { 0, 1 };
	const int histSize[] = { 64, 64 };
	float range[] = { 0, 256 };
	const float *ranges[] = { range, range };

	// Histogram in object region
	Mat objectROI = frame(object_region);
	calcHist(&objectROI, 1, channels, noArray(), objectHistogram, 2, histSize, ranges, true, false);


	// A priori color distribution with cumulative histogram
	calcHist(&frame, 1, channels, noArray(), globalHistogram, 2, histSize, ranges, true, true);


	// Boosting: Divide conditional probabilities in object area by a priori probabilities of colors
	for (int y = 0; y < objectHistogram.rows; y++) {
		for (int x = 0; x < objectHistogram.cols; x++) {
			objectHistogram.at<float>(y, x) /= globalHistogram.at<float>(y, x);
		}
	}
	normalize(objectHistogram, objectHistogram, 0, 255, NORM_MINMAX);
}
void backProjection(const Mat &frame, const Mat &histogram, Mat &bp) {
	const int channels[] = { 0, 1 };
	float range[] = { 0, 256 };
	const float *ranges[] = { range, range };
	calcBackProject(&frame, 1, channels, objectHistogram, bp, ranges);
}
/*int main()
{
	

	CallMouseInfo param;
	getHistogram getHist;

	bool whatTracking = false;

	//IplImage *frame;
	cv::VideoCapture vc(0);
	cv::Mat frame;
	Size size = Size((int)vc.get(CAP_PROP_FRAME_WIDTH),(int)vc.get(CAP_PROP_FRAME_HEIGHT));
	cv::Mat before[3]; 
	IplImage *difflabel= cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage *image = cvCreateImage(size, IPL_DEPTH_8U, 3);
	IplImage *frame2 = cvCreateImage(size, IPL_DEPTH_8U, 3);
	//InitFrameBuffer(before, size);
	int last = 0;
	int t = 0;
	int curr, prev;
	Mat img;
	Mat diff;
	bool cam = false;
	Mat model, back, hue[3];
	Mat mask;// hist 범위
	Mat result;// hist 결과 저장
	Mat hsv;
	int LowH = 170;
	int HighH = 179;

	int LowS = 50;
	int HighS = 255;

	int LowV = 0;
	int HighV = 255;

	float hsvRange[2] = { 0, 180 };
	const float* ranges = { hsvRange };
	int count = 0;
	diff.create(size, CV_8UC1);
	img.create(size, CV_8UC3);
	hsv.create(size, CV_8UC3);
	
	Mat hu1;
	
	hue[0].create(size, CV_8UC1);
	hue[1].create(size, CV_8UC1);
	hue[2].create(size, CV_8UC1);
	hu1.create(size, CV_8UC1);
	
	clock_t begin, end;
	CBlobLabeling label = CBlobLabeling();//라벨링을 위한 객체 생성
	cv::Rect rc;
	int histSize = 16;
	
	int smin = 30, vmin = 10, vmax = 255;
	bool track = false;
	if (!vc.isOpened()) {
		std::cout << "can not open the video !! zzzzzzzzzzzzzzz" << std::endl;
		return -1;
	}

	vc.read(frame);

	param.m_frame = frame;

	//cv::setMouseCallback("display", onMouse, &param);
	begin = clock();
	while (1)
	{
		//vc.read(frame);
		vc >> frame;
		img = frame;
		cvtColor(img, hsv, COLOR_BGR2HSV);
		//cv::imshow("display", frame);
		frame2 = &IplImage(frame);
		cvCopy(frame2, image, 0);
		cvtColor(frame, before[last], CV_BGR2GRAY);
		curr = last;
		prev = (curr + 1) % 3;
		last = prev;
		if (t < 3) {
			t++;
			continue;
		}

		absdiff(before[prev], before[curr], diff);
		threshold(diff, diff, 10, 255, CV_THRESH_BINARY);
		//cv::imshow("display",diff);
		Mat element(7, 7, CV_8U, cv::Scalar(1));
		erode(diff, diff, element);
		dilate(diff, diff, element);
		
		difflabel = &IplImage(diff);

			end = clock();
			if (track) {
				//getHist.go_histogram(img, rc);
				//getHist.go_tracking(img, rc);
				
			}
			cout << (end - begin) / CLOCKS_PER_SEC << endl;
			if ((end-begin)/CLOCKS_PER_SEC>=1.0) {
				track = true;
			}
			if (track) {
				rc = label.DectRectangle(difflabel, image);
				img = cvarrToMat(image);
				mask = img;
				//param.m_roi = rc;
				
				inRange(hsv, Scalar(LowH, LowS, LowV), Scalar(HighH, HighS, HighV), hu1);
				
				MatND histogram;
				const int* channel_numbers = { 0 };
				float channel_range[] = { 0.0, 255.0 };
				const float* channel_ranges = channel_range;
				int number_bins = 255;

				calcHist(&hu1, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

				// Plot the histogram
				int hist_w = 512; int hist_h = 400;
				int bin_w = cvRound((double)hist_w / number_bins);

				Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
				normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());
				for (int i = 1; i < number_bins; i++)
				{
					line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
						Point(bin_w*(i), hist_h - cvRound(histogram.at<float>(i))),
						Scalar(255, 0, 0), 2, 8, 0);
				}


		

			
				//getObjectHistogram(img, rc);
				
				param.hist_ready = true;
				track = false;
			}
			
			/*if (param.hist_ready)
			{
			cv::Rect rc1 = param.m_roi;
			getHist.go_histogram(frame, rc1);

			param.hist_ready = false;
			whatTracking = true;
			}*/



			/*if (whatTracking)
			{
			cv::Rect rc;
			getHist.go_tracking(frame, rc);
			param.hist_ready = true;
			}

			if (cam) {
				//getHist.go_tracking(img, rc);
				//getHist.go_histogram(img, rc);
				
				
				
				//CvScalar(0, 255, 0)
				cvtColor(img, hsv, COLOR_BGR2HSV);//smin 30 vmin 10 vmax 255
				inRange(hsv, cv::Scalar(0, smin, vmin,0),  // 범위 안에 들어가면 0으로 만들고 나머지 1로 만들어 gray 이미지로 만듦
					cv::Scalar(180, 255, vmax,0), mask);// mask에 결과가 저장됨 => gray 이미지??
				

															 // 동영상으로 들어오는 프레임을 count해서 일정 프레임 지나고 나면 해당 영역을 camshift로 tracking하게 하기 //count가 안됨...왜???
															 // hue값에 label된 이미지 영역 제대로 넣기
															 //m=camshift 인자값												 // 각 함수 인자 제대로 설정 이해하기
				int ch[] = { 0, 0 };
				// hue.create(hsv.size(), hsv.depth());
				//mixChannels(&hsv, 1, &hue, 3, ch, 1);// 각 채널로 분할
				split(hsv, hue);
				//Mat roi(hue[0], rc);
				//Mat maskroi(mask, rc);
				Mat roi(hue[0], rc), maskroi(mask, rc); // 각 지정 영역만큼만 저장

				calcHist(&roi, 1, 0, maskroi, model, 1, &histSize, &ranges);
				//계산할 이미지, image배열에 포함된 이미지 개수 , 계산할 채널 번호들 배열,계산할 영역 지정(maskroi=0이면 동작 안함),
				//model=계산 결과 저장, 1=hist 차원 가리킴, histSize 각 차원 사이즈 , ranges 각 차원 최소값과 최대값
				normalize(model, result, 0, 255, NORM_MINMAX);
				calcBackProject(&hue[0], 1, 0, result, back, &ranges);
				back &= mask;

				RotatedRect trackBox = CamShift(back, rc, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));

				ellipse(img, trackBox, CV_RGB(0, 0, 0), 3, LINE_AA);
				
			/*************************************여기서 부터 안돌아감***********************************
			int ch[] = { 0, 0 };
				// hue.create(hsv.size(), hsv.depth());
				//mixChannels(&hsv, 1, &hue, 3, ch, 1);// 각 채널로 분할
				split(hsv, hue);
				Mat roi(hue[0], rc), maskroi(mask, rc); // 각 지정 영역만큼만 저장
			
				calcHist(&roi, 1, 0, maskroi, model, 1, &histSize, &ranges);
				//계산할 이미지, image배열에 포함된 이미지 개수 , 계산할 채널 번호들 배열,계산할 영역 지정(maskroi=0이면 동작 안함),
				//model=계산 결과 저장, 1=hist 차원 가리킴, histSize 각 차원 사이즈 , ranges 각 차원 최소값과 최대값
				normalize(model, result, 0, 255, NORM_MINMAX);
				calcBackProject(&hue[0], 1, 0, result, back, &ranges);
				back &= mask;

				RotatedRect trackBox = CamShift(back, rc, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));

				ellipse(img, trackBox, CV_RGB(0, 0, 0), 3, LINE_AA);
				
			}
		
			
		cv::imshow("display", img);


			if (cv::waitKey(30) == 27) {
				break;
			}
			

	}
	
	
	cvReleaseImage(&image);
	label.~CBlobLabeling();
	return 0;
}*/


int main()
{
	cv::VideoCapture vc(0);
	//Mat srcImage = imread("fruits.jpg");
	//CvCapture *capture = cvCaptureFromAVI("sam1.avi");
	
	//VideoCapture vc1("sam1.avi");

	CBlobLabeling label = CBlobLabeling();//라벨링을 위한 객체 생성
	getHistogram getHist;
	cv::Rect rc;
	int last = 0;
	int t = 0;
	int curr, prev;
	Mat img;
	Mat diff;
	cv::Mat before[3];
	bool labeling = false;
	bool track = false;
	Size size = Size((int)vc.get(CAP_PROP_FRAME_WIDTH), (int)vc.get(CAP_PROP_FRAME_HEIGHT));
	//Size size = Size((int)vc1.get(CAP_PROP_FRAME_WIDTH), (int)vc1.get(CAP_PROP_FRAME_HEIGHT));
	IplImage *difflabel = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage *image = cvCreateImage(size, IPL_DEPTH_8U, 3);
	IplImage *roi = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage *maskroi = cvCreateImage(size, IPL_DEPTH_8U, 1);
	CvRect rect;
	Rect rect1;
	Mat roiImage;
	Mat roim;
	Mat hueImage;
	//const int channel[] = {0,1};
	const int *channel = { 0 };
	//int histSize = 256;
	//float  hValue[] = { 0, 256 };
	int histSize = 255;
	float  hValue[] = { 0.0, 255.0 };
	//const  float* ranges[] = { hValue };
	const  float* ranges = hValue;
	Mat hist;
	int smin = 30, vmin = 10, vmax = 255;
	Mat mask, model;
	MatND model1;
	vector<Mat> planes;
	roiImage.create(size, CV_8UC1);
	clock_t begin, end;
	/*if (!vc.isOpened()) {
		std::cout << "can not open the video !! zzzzzzzzzzzzzzz" << std::endl;
		return -1;
	}*/
	if (!vc.isOpened()) {
		std::cout << "The video file was not found" << std::endl;
		return 0;
	}
	Mat frame;
	Mat hsvImage;
	while (1)
	{
		//vc.read(frame);
		//vc >> frame;//실시간 영상처리일 경우
		begin= clock();
		vc >> frame;
		
		image = &IplImage(frame);
		
		//cvtColor(frame, hsvImage, COLOR_BGR2HSV); // HSV 영상으로 변경

		cvtColor(frame, before[last], CV_BGR2GRAY);
		curr = last;
		prev = (curr + 1) % 3;
		last = prev;
		if (t < 3) {
			t++;
			continue;
		}

		absdiff(before[prev], before[curr], diff);
		threshold(diff, diff, 10, 255, CV_THRESH_BINARY);
		
		Mat element(7, 7, CV_8U, cv::Scalar(1));
		erode(diff, diff, element);
		
		dilate(diff, diff, element);

		difflabel = &IplImage(diff);
		
		end = clock();
		if (!labeling) {
			rc = label.DectRectangle(difflabel, image);
			img = cvarrToMat(image);
			//cv::imshow("label", img);
			/*if ((end - begin) / CLOCKS_PER_SEC >= 1.0) {
				track = true;
				labeling = true;
			}*/
			//labeling = true;
			track = true;
			//labeling = true;
		}
		//imshow("srcImage", img);
		if (track) {
			cvtColor(img, hsvImage, COLOR_BGR2HSV); // HSV 영상으로 변경
			split(hsvImage, planes); // 채널 분리  
			hueImage = planes[0]; // planes[0] = Hue 색상

			rect.x = rc.x;
			rect.y = rc.y;
			rect.width = rc.width;
			rect.height = rc.height;

			rect1 = rc;

			roi = &IplImage(hueImage);
			//imshow("mask", hueImage);

			//cvZero(roi);//0으로 초기화

			/*for (int h1 = rect.y; h1 <= rect.y + rect.height;h++) {
				for (int w1 = rect.x; w1<= rect.x + rect.width;w++ ) {
					//roi->imageData[w + h * (roi->widthStep)] = 0;
					cvSet2D(roi, h1, w1, cvScalar(0));

				}
			}*/

			/*for (int h = 0; h <roi->height; h++) {
				for (int w = 0; w < roi->width; w++) {
					if ((rect.x <= w && w <= rect.width) && (h >= rect.y && h <= rect.height)) {
						//maskroi->imageData[w + h* (roi->widthStep)] = 1;

					}
					else {
						roi->imageData[w + h * (roi->widthStep)] = 0;
					}

				}
			}*/
			
			cvSetImageROI(roi, rect);
			roiImage = cvarrToMat(roi);
			//imshow("mask", roiImage);

			cv::inRange(hsvImage, cv::Scalar(0, smin, MIN(vmin, vmax)),
				cv::Scalar(180, 256, MAX(vmin, vmax)), mask);
			imshow("mask", mask);
			maskroi = &IplImage(mask);

			/*for (int h1 = rect.y; h1 <= rect.y + rect.height; h1++) {
				for (int w1 = rect.x; w1 <= rect.x + rect.width; w1++) {
					maskroi->imageData[w1+ h1* (roi->widthStep)] = 0;


				}
			}*/

			/*for (int h = 0; h <maskroi->height;h++) {
				for (int w = 0; w < maskroi->width;w++) {
					if ((rect.x<=w  && w <= rect.width )&&  (h>=rect.y && h<=rect.height)) {
						

					}
					else {
						maskroi->imageData[w + h * (maskroi->widthStep)] = 0;
					}

				}
			}
			cvShowImage("maskroi", maskroi);*/

			cvSetImageROI(maskroi, rect);
			roim = cvarrToMat(maskroi);
			//imshow("roim", roim);
			//PlaySound(TEXT("SOUND_FILE_NAME"), NULL, SND_FILENAME | SND_SYNC);

			Mat backProject; // 역투영
			//cv::calcHist(&roiImage, 1, 0, roim, model, 1, &histSize, ranges, true, false);
			calcHist(&roiImage, 1, channel, roim, model1, 1, &histSize, &ranges, true, false);
			cv::normalize(model1, model1, 0, 255, cv::NORM_MINMAX);

			/*for (int i = 0; i < 20;i++) {
				cv::calcBackProject(&roiImage, 1, channel, model1, backProject, &ranges);
				backProject = mask;

				RotatedRect trackBox = cv::CamShift(backProject, rc, cvTermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 1));

				cv::ellipse(img, trackBox, cv::Scalar(0, 255, 255), 3, CV_AA);
			}*/

			cv::calcBackProject(&roiImage, 1, channel, model1, backProject, &ranges);
			


			backProject = mask;
			//bitwise_and(backProject, roim, backProject);
			//imshow("backProject", backProject);
			//RotatedRect trackBox = cv::CamShift(backProject, rc, cvTermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20,1));
			RotatedRect trackBox = CamShift(backProject, rect1, cvTermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 1));
			//trackBox.center.x =( rect1.x + rect1.width )/ 2;
			//cout<<(rect1.x + rect1.width) / 2<<endl;
			cv::ellipse(img, trackBox, cv::Scalar(0, 255, 255), 3, CV_AA);
			//labeling = false;
			//track = false;
			//labeling = false;

			//calcBackProject(&hueImage, 1, 0, hist, backProject, ranges);
			//cv::calcBackProject(&hueImage, 1, 0, model, backProject, ranges);
			//imshow("roim", model1);

			//getHist.go_tracking(frame, rc);
			/*Mat backProject; // 역투영
			/***********************************error*********mask가 문제***************************************
			cv::calcHist(&roiImage, 1, 0, roim, model, 1, &histSize, ranges, true, false);
			/**********************************************************************************
			cv::normalize(model, model, 0, 255, cv::NORM_MINMAX);
			cv::calcBackProject(&hueImage, 1, 0, model, backProject, ranges);

			backProject &= mask;

			cv::RotatedRect trackBox = cv::CamShift(backProject, rc, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
			cv::ellipse(img, trackBox, cv::Scalar(0, 255, 255), 3, cv::LINE_AA);
			*/

			//cv::imshow("back", roiImage);
			//calcHist(&roiImage, 1, 0, Mat(), hist, 1, &histSize, ranges);


			//calcBackProject(&roiImage, 1, 0, hist, backProject, ranges);

			//Mat backProject; // 역투영
			/*double minVal, maxVal;
			minMaxLoc(backProject, &minVal, &maxVal);*/



			//backProject를 범위[0,255]로 정규화해서 backProject2에 저장
//			Mat backProject2;
			//cv::imshow("back", backProject);

			/**********되는 부분*******/
			//normalize(backProject, backProject2, 0, 255, NORM_MINMAX, CV_8UC1);
			//normalize(backProject, backProject2, 0, 255, NORM_MINMAX, CV_8U);
			/**********되는 부분*******/

			/*cv::Mat back_frame;
			cv::cvtColor(backProject, back_frame, cv::COLOR_GRAY2BGR);
			cv::imshow("back", back_frame);
			*/



		}


		imshow("srcImage", img);
		//imshow("srcImage",roiImage);
		/*
		vector<Mat> planes;
		split(hsvImage, planes); // 채널 분리
		Mat hueImage = planes[0]; // planes[0] = Hue 색상

								  //    Rect roi(100, 100, 100, 100); // yellow orange
		//Rect roi(400, 150, 100, 100); // green kiwi 관심영역 설정
		rectangle(frame, rc, Scalar(0, 0, 255), 2); // 빨간색으로 사각형 그리기
		Mat roiImage = hueImage(rc); // roiImage = 관심영역

		int histSize = 256;
		float  hValue[] = { 0, 256 };
		const  float* ranges[] = { hValue };
		Mat hist;
		calcHist(&roiImage, 1, 0, Mat(), hist, 1, &histSize, ranges);

		Mat backProject; // 역투영
		calcBackProject(&hueImage, 1, 0, hist, backProject, ranges);

		double minVal, maxVal;
		minMaxLoc(backProject, &minVal, &maxVal);
		cout << "minVal=" << minVal << endl;
		cout << "maxVal=" << maxVal << endl;

		//backProject를 범위[0,255]로 정규화해서 backProject2에 저장
		Mat backProject2;
		normalize(backProject, backProject2, 0, 255, NORM_MINMAX, CV_8U);

		imshow("backProject2", backProject2);
		imshow("srcImage", frame);
		*/
		if (cv::waitKey(30) == 27) {
			break;
		}
	}
	cvReleaseImage(&roi);
	cvReleaseImage(&maskroi);
	cvReleaseImage(&difflabel);
	cvReleaseImage(&image);
	return 0;
}