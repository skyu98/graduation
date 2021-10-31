#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include <chrono>

using namespace std;
using namespace cv;

int main() {
	// Mat image = imread("./smallScale.jpg");
	Mat image = imread("./test.jpg");
	resize(image, image, Size(900, 500));
	imshow("image", image);

	Mat mask = Mat::zeros(image.size(), CV_8UC1);
	Rect rect = selectROI("image", image, false);
	imwrite("./withRect.jpg", image);
	destroyWindow("image");
	Mat bgdModel, fgdModel;

	auto start = std::chrono::high_resolution_clock::now();
	grabCut(image, mask, rect, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> timeUsed = end - start;	// ����

	cout << "Time Of GrabCut is : " << timeUsed.count() << endl;

	start = std::chrono::high_resolution_clock::now();
	Mat result = Mat::zeros(image.size(), CV_8UC3);
	for (int row = 0; row < result.rows; row++)
	{
		for (int col = 0; col < result.cols; col++)
		{
			//�����Ĥmask��ĳ��λ��������ֵΪ1��3��Ҳ��������ǰ���Ϳ���ǰ�����Ͱ�ԭͼ���и�λ�õ�����ֵ�������ͼ��
			if (mask.at<uchar>(row, col) == 1 || mask.at<uchar>(row, col) == 3)
			{
				result.at<Vec3b>(row, col) = image.at<Vec3b>(row, col);
			}
		}
	}
	end = std::chrono::high_resolution_clock::now();

	timeUsed = end - start;	// ����

	cout << "Time Of Using Mask is : " << timeUsed.count() << endl;

	imshow("result", result);
	// imwrite("./output_smallScale.jpg", result);
	imwrite("./output.jpg", result);
	waitKey(0);
}


