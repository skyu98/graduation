#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include "utils.h"

using namespace cv;
using namespace std;

string input_dir = "../imgs/input_imgs/";
string output_dir = "../imgs/output_imgs/";

#define ORIGINAL 0
#define DIFF 1

void getOriginalContourFromGray(Mat& src, int gray_method = DIFF) {
    // 均值降噪
    Mat img;
    cv::GaussianBlur(src, img, Size(11, 11), 0, 0);

    Mat gray;
    if(gray_method == ORIGINAL) {
        // 传统灰度转化方式
        cvtColor(img, gray, CV_RGB2GRAY);
        imshow("gray_original", gray);
    }
    else if(gray_method == DIFF) {
        // 通道分割方式
        Mat channels[3];
        split(img, channels);
        // imshow("B", channels[0]);
        // imshow("G", channels[1]);
        // imshow("R", channels[2]);

        Mat diff;
        absdiff(channels[2], channels[0], diff);
        my_utils::diff(channels[2], diff, channels[2], 4);
        gray = std::move(channels[2]);
        imshow("gray_diff", gray);
    }

    // 二值化
    Mat binaryImg;
    // 0 - black 255 - white
    cv::threshold(gray, binaryImg, 45, 255, THRESH_BINARY);

    Mat paddedImg;
    cv::copyMakeBorder(binaryImg, paddedImg, 1, 1, 1, 1, BORDER_CONSTANT, 255);

    // 获取轮廓
    vector<vector<Point> > contours;
    vector<Vec4i> hireachy;
    cv::findContours(paddedImg, contours, hireachy, CV_RETR_LIST, CHAIN_APPROX_SIMPLE, Point());

    // 获取最大轮廓
    for (size_t t = 0; t < contours.size(); ++t) {
        Rect rect = boundingRect(contours[t]);
        if( rect.width < src.cols / 2) continue;

        drawContours(src, contours, t, CV_RGB(255, 0, 0), 2, 8);
        break;
    } 
}

int main(int argc, char* argv[]) {

    string imgName = argc >= 2 ? argv[1] : "1.jpg";
    Mat src = imread(input_dir + imgName);
    Mat src2 = src.clone();

    // getOriginalContourFromGray(src, ORIGINAL);
    getOriginalContourFromGray(src2, DIFF);

    // imshow("res_original", src);
    imshow("res_diff", src2);
   
    waitKey(0);
    return 0;
}