#include "utils.h"

namespace my_utils {

Mat rotateImage(const Mat& image, double angle) {
	Mat dst, M;
	int h = image.rows;
	int w = image.cols;
    
    Point center(image.size().width / 2, image.size().height / 2);
	M = getRotationMatrix2D(center, -1 * angle, 1.0);

	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));

	int new_w = cos * w + sin * h;
	int new_h = cos * h + sin * w;
	M.at<double>(0, 2) += (new_w / 2.0 - w / 2);
	M.at<double>(1, 2) += (new_h / 2.0 - h / 2);
	warpAffine(image, dst, M, Size(new_w, new_h), INTER_LINEAR, 0, Scalar(255, 255, 0));
	return dst;
}

Point getRotatedPoint(Point inputPoint,Point center,double angle) {
    Point rotatedPoint;
    rotatedPoint.x = (inputPoint.x - center.x) * cos(-1 * angle) - (inputPoint.y - center.y) * sin(-1 * angle) + center.x;
    rotatedPoint.y = (inputPoint.x - center.x) * sin(-1 * angle) + (inputPoint.y - center.y) * cos(-1 * angle) + center.y;
    return rotatedPoint;
}

void putText(Mat& img, const string& text, Point textPos) {
	Scalar textColor(0,0,255);
	cv::putText(img, text.c_str(), textPos, CV_FONT_HERSHEY_SIMPLEX, 1, textColor, 2);
}

void morphology(const Mat &imgIn, Mat &imgOut, int morpOp,
                int minThickess, int shape, int iter) {
    int size = minThickess / 2;
    Point anchor = Point(size, size);
    Mat element = getStructuringElement(shape, Size(2 * size + 1, 2 * size + 1), anchor);
    morphologyEx(imgIn, imgOut, morpOp, element, anchor, iter);
}

void diff(Mat& src, const Mat& mask, Mat& dst, int iterations) {
    if(iterations <= 0) return;
    for(int i = 0;i < iterations;++i) {
        cv::absdiff(src, mask, dst);
    } 
}


}
