#include "findContour.h"

vector<cv::Point>& getMaxContour(const string& srcImgName, vector<cv::Point>& contour) {
    cv::Mat srcImg;
    srcImg = cv::imread(srcImgName);
    if (srcImg.empty()) {
        printf("colud not load image ..\n");
        return {};
    }

    // 均值降噪
    cv::Mat blurImg;
    cv::GaussianBlur(srcImg, blurImg, cv::Size(7, 7), 0, 0);

    // 灰度图
    cv::Mat grayImg, binaryImg;
    cv::cvtColor(blurImg, grayImg, cv::COLOR_BGR2GRAY);
    imshow("grayImg", grayImg);
   
    // 二值化
    cv::Mat binaryImg;
    cv::threshold(grayImg, binaryImg, 60, 255, cv::THRESH_BINARY);

    cv::Mat paddedImg;
    copyMakeBorder(binaryImg, paddedImg, 10, 10, 10, 10, cv::BORDER_CONSTANT, 255);

    // 获取轮廓
    cv::Mat resImg = cv::Mat::zeros(srcImg.size(), CV_8UC3);
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hireachy;
    findContours(paddedImg, contours, hireachy, CV_RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point());

    // 获取最大轮廓
    for (size_t t = 0; t < contours.size(); t++) {
        cv::Rect rect = boundingRect(contours[t]);
        if( rect.width < srcImg.cols / 2) continue;

        contour = std::move(contours[t]);
        break;
        // double area = contourArea(contours[t]);
        // double len = arcLength(contours[t], true);
       
        // printf("area of star could : %f \n", area);
        // printf("lenght of star could : %f \n", len);
    } 
    return contour;
}
