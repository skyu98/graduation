#ifndef __GETCONTOUR_H__
#define __GETCONTOUR_H__

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;

void getBiggestContour(const Mat& srcImgName, vector<Point>& contour);

#endif // __GETCONTOUR_H__