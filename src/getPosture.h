#ifndef __GETPOSTURE_H__
#define __GETPOSTURE_H__

#include "findContour.h"

typedef struct {
    double x;
    double y;
    double theta;
    double width;
    double height; 
} Posture;

Posture getPosture(const string& srcImgName, const vector<cv::Point>& contour);

#endif // __GETPOSTURE_H__