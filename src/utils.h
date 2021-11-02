#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <string>

using namespace std;
using namespace cv;

namespace my_utils {
void ImageStretchByHistogram(const Mat& src, Mat & dst);

/* 获得单个点经过旋转后所在精确坐标
* @param angle : 顺时针为正，逆时针为负 
*/
Point getRotatedPoint(Point inputPoint,Point center,double angle); 

void putText(Mat& img, const string& text, Point textPos);
}

#endif // UTILS_H