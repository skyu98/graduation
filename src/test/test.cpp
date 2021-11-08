#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include "utils.h"

using namespace cv;
using namespace std;



void Main_Inpaint() {
    Mat src,blur,mask,bkMask,fgMask,dst;
    vector<vector<Point> > contours;
    src = imread("../imgs/input_imgs/2.jpg");
    // remove noise
    cv::GaussianBlur(src,blur,Size(),2,2);
    //CREATE A MASK FOR THE SATURATED PIXEL
    int minBrightness=100;
    int dilateSize=20;
    //convert to HSV
    Mat src_hsv,brightness,saturation;
    vector<Mat> hsv_planes;
    cvtColor(blur, src_hsv, COLOR_BGR2HSV);
    split(src_hsv, hsv_planes);
    brightness = hsv_planes[2];
    //get the mask
    threshold(brightness,mask,minBrightness,255,THRESH_BINARY);
    imshow("mask", mask);
    //dialte a bit the selection
    my_utils::morphology(mask,mask,MORPH_DILATE,dilateSize);
    //INPAINTING
    float radius=5.0;

    // inpaint(src,mask,dst,radius,INPAINT_TELEA);
    // imshow("Method by Alexandru Telea ",dst);
    //show the selection on src
    findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++)
        drawContours(src,contours,i,Scalar(0,0,255),2);
    imshow("Inpaint mask",src);
    waitKey(0);
}

string input_dir = "../imgs/input_imgs/";
string output_dir = "../imgs/output_imgs/";

int main(int argc, char* argv[]) {
    // Main_Inpaint();
    string imgName = argc >= 2 ? argv[1] : "1.jpg";
    Mat img = imread(input_dir + imgName);
    int row = img.size().height, col = img.size().width;

    Mat channels[3];
    split(img, channels);
    imshow("0", channels[0]);
    // imshow("1", channels[1]);
    // imshow("2", channels[2]);

    Mat diff;
    absdiff(channels[2], channels[0], diff);

    my_utils::diff(channels[2], diff, channels[2], 2);
    imshow("res", channels[2]);

    // my_utils::rotateImage(img, 45);

    // Mat blur;
    // cv::GaussianBlur(img,blur,Size(),2,2);

    // Mat gray;
    // cvtColor(blur, gray, COLOR_BGR2GRAY);

    // Mat edges;
    // Canny(gray, edges, 30, 60, 3, true);
    // imshow("edges", edges);

    // vector<Vec4i> plines;
    // HoughLinesP(edges, plines, 1, CV_PI/180, 150, 10, 10);
    // for(size_t i =0; i< plines.size(); i++)
    // {
    //     Vec4i points = plines[i];
    //     line(img, Point(points[0], points[1]), Point(points[2],points[3]), Scalar(0,255,255), 3, CV_AA);
    // }    

    // Mat frame_hsv;
    // cvtColor(img, frame_hsv, COLOR_BGR2HSV);
    // imshow("hsv", frame_hsv);
    // Mat res = Mat::zeros(img.size(), CV_8UC3);

    // vector<int> l = {23, 0, 0};
    // vector<int> u = {80, 105, 107};
    // InputArray lower(l), upper(u);

    // Mat mask;
    // inRange(frame_hsv, l, u, mask);
    // imshow("mask", mask);

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
    

    // imshow("img", img);
    // // imshow("res", res);
    waitKey(0);
    return 0;
}