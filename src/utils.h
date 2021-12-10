#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <sys/stat.h>
#include <dirent.h>
#include <string>
#include <stdio.h>
#include <string.h>

using namespace std;
using namespace cv;

/* Image */
namespace my_utils {

void ImageStretchByHistogram(const Mat& src, Mat & dst);
Point getRotatedPoint(const Point& inputPoint, const Point& center, double angle); 

#define FULL_IMG 0
#define CUTTED_IMG 1
Mat rotateImage(const Mat& image, double angle, int flag = FULL_IMG);
void morphology(Mat &imgIn, Mat &imgOut, int morpOp = MORPH_CLOSE,
                int minThickess = 2,int shape = MORPH_ELLIPSE, int iter = 1);
void diff(Mat& src, const Mat& mask, Mat& dst, int iterations = 1);
void putText(Mat& img, const string& text, Point textPos);

/* Data Analysis */
void getKDE(const vector<int>& x_array, const vector<int>& data,  vector<double>& y, double bandwidth = 5);

/* File and Text */
void traverseFolder(const char* pInputPath, std::vector<string>& fileNames);
void writeResultToFile(const FILE* filePath, const char* msg);
void splitStrToVec(const string& str, const string& split, vector<string>& vec);
}

#endif // UTILS_H