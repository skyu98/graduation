#include "getContour.h"

void getBiggestContour(const Mat& srcImg, vector<Point>& contour) {
    // 均值降噪
    Mat blurImg;
    cv::GaussianBlur(srcImg, blurImg, Size(7, 7), 0, 0);

    // 灰度图
    Mat grayImg, binaryImg;
    cv::cvtColor(blurImg, grayImg, COLOR_BGR2GRAY);
    imshow("grayImg", grayImg);
   
    // 二值化
    Mat binaryImg;
    cv::threshold(grayImg, binaryImg, 60, 255, THRESH_BINARY);

    Mat paddedImg;
    cv::copyMakeBorder(binaryImg, paddedImg, 10, 10, 10, 10, BORDER_CONSTANT, 255);

    // 获取轮廓
    Mat resImg = Mat::zeros(srcImg.size(), CV_8UC3);
    vector<vector<Point>> contours;
    vector<Vec4i> hireachy;
    cv::findContours(paddedImg, contours, hireachy, CV_RETR_LIST, CHAIN_APPROX_SIMPLE, Point());

    // 获取最大轮廓
    for (size_t t = 0; t < contours.size(); t++) {
        Rect rect = boundingRect(contours[t]);
        if( rect.width < srcImg.cols / 2) continue;

        contour = std::move(contours[t]);
        break;
    } 
}
