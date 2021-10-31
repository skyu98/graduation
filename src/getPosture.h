#ifndef __GETPOSTURE_H__
#define __GETPOSTURE_H__

#include "getContour.h"

typedef struct {
    Point2f center = Point2f(0, 0);
    double angle = 0;
    double width = 0;
    double height = 0; 
} Posture;

Posture getPosture(const vector<Point>& contour, Mat& srcImg, bool drawResult = false);
void getgetOrientation(const vector<Point>& contour, Posture& posture, Mat& srcImg, bool drawResult = false);
void getSize(const vector<Point>& contour, Posture& posture, Mat& srcImg, bool drawResult = false);

#endif // __GETPOSTURE_H__