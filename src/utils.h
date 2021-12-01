#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <string>

using namespace std;
using namespace cv;

//helper functions
namespace my_utils {
void ImageStretchByHistogram(const Mat& src, Mat & dst);

/** @brief 获得单个点经过旋转后所在精确坐标
    @param inputPoint 需要旋转的点
    @param angle 顺时针为正，逆时针为负 
*/
Point getRotatedPoint(Point inputPoint, Point center, double angle); 

/** @brief 绕中心旋转图片并保持其原尺寸
    @param angle 顺时针为正，逆时针为负 
*/
#define FULL_IMG 0
#define CUTTED_IMG 1
Mat rotateImage(const Mat& image, double angle, int flag = FULL_IMG);


void splitStrToVec(const string& str, const string& split, vector<string>& vec);
void putText(Mat& img, const string& text, Point textPos);

void morphology(Mat &imgIn, Mat &imgOut, int morpOp = MORPH_CLOSE,
                int minThickess=2,int shape = MORPH_ELLIPSE, int iter = 1);

void diff(Mat& src, const Mat& mask, Mat& dst, int iterations = 1);

void getKDE(const vector<int>& x_array, const vector<int>& data,  vector<double>& y, double bandwidth = 5);
}

#endif // UTILS_H