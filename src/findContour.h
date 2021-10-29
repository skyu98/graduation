#ifndef __FINDCONTOUR_H__
#define __FINDCONTOUR_H__

#include <opencv2/opencv.hpp>
using namespace std;

vector<cv::Point>& getMaxContour(const string& srcImgName, vector<cv::Point>& contour);

#endif // __FINDCONTOUR_H__