#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    Mat img = imread("../imgs/cv_imgs/01.jpg");
    int row = img.size().height, col = img.size().width;

    Mat frame_hsv;
    cvtColor(img, frame_hsv, COLOR_BGR2HSV);
    
    Mat res = Mat::zeros(img.size(), CV_8UC3);

    vector<int> l = {23, 0, 0};
    vector<int> u = {80, 105, 107};
    InputArray lower(l), upper(u);

    Mat mask;
    inRange(frame_hsv, l, u, mask);
    imshow("mask", mask);

    // for(int x = 0;x < col;++x) {
    //     for(int y = 0;y < row;++y) {
    //         if((static_cast<int>(img.at<cv::Vec3b>(y, x)[0] +
    //             static_cast<int>(img.at<cv::Vec3b>(y, x)[1]) +
    //             static_cast<int>(img.at<cv::Vec3b>(y, x)[2])) < 120)) {
    //                 res.at<cv::Vec3b>(y, x)[0] = 255;
    //                 res.at<cv::Vec3b>(y, x)[1] = 255;
    //                 res.at<cv::Vec3b>(y, x)[2] = 255;
    //             }

    //         // cout << static_cast<int>(img.at<cv::Vec3b>(y, x)[0]) << endl;
    //         // cout << static_cast<int>(img.at<cv::Vec3b>(y, x)[1]) << endl;
    //         // cout << static_cast<int>(img.at<cv::Vec3b>(y, x)[2]) << endl;
    //     }
    // }
    

    imshow("img", img);
    imshow("res", res);
    waitKey(0);
    return 0;
}