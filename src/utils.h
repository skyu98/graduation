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

/* 获得单个点经过旋转后所在精确坐标
* @param angle : 顺时针为正，逆时针为负 
*/
Point getRotatedPoint(Point inputPoint,Point center,double angle); 

Mat rotateImage(const Mat& image, double angle);

void putText(Mat& img, const string& text, Point textPos);

void morphology(const Mat &imgIn, Mat &imgOut, int morpOp = MORPH_CLOSE,
                int minThickess=2,int shape = MORPH_ELLIPSE, int iter = 1);

void diff(Mat& src, const Mat& mask, Mat& dst, int iterations = 1);

}



#endif // UTILS_H