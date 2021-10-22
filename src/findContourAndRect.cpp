#include <opencv2/opencv.hpp>
#include <iostream>
#include <direct.h>
#include <math.h>
#include <string>
 
using namespace std;
using namespace cv;

// 0 - black 255 - white

Mat src, dst;
string input_dir = "../test_imgs/";
string output_dir = "../output_imgs/";
string input_num = "1";

char output_image[] = "output image";

void fillHole(const Mat srcBw, Mat& dstBw) {
    Size m_Size = srcBw.size();
    Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());//延展图像
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

    cv::floodFill(Temp, Point(0, 0), Scalar(255));

    Mat cutImg;//裁剪延展的图像
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

    dstBw = srcBw | (~cutImg);
}

void opposite(Mat& src) {
    int rows = src.rows, cols = src.cols;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            int px_value = src.at<uchar>(i, j);
            //cout << px_value<< endl;
            src.at<uchar>(i, j) = 255 - px_value;
        }
    }
}

int main(int argc, char argv[]) {


    src = imread(input_dir + input_num + "/" + input_num + ".jpg");
    if (src.empty()) {
        printf("colud not load image ..\n");
        return -1;
    }

    // namedWindow(input_image, CV_WINDOW_NORMAL);
    // namedWindow(output_image, CV_WINDOW_NORMAL);
    // imshow(input_image, src);

    // 均值降噪
    Mat blurImg;
    GaussianBlur(src, blurImg, Size(7, 7), 0, 0);
    // imshow("blured", src);

    // 灰度图二值化
    Mat gray_src, binary;
    cvtColor(blurImg, gray_src, COLOR_BGR2GRAY);
    threshold(gray_src, binary, 68, 255, THRESH_BINARY);
    // imshow("binary", binary);
    imwrite(output_dir + input_num + "/binary.jpg", binary);

   
    Mat morphImage;
    Mat bigKernel = getStructuringElement(MORPH_RECT, Size(15, 15));
    Mat smallKernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat horizonal = getStructuringElement(MORPH_RECT, Size(15, 1));
    Mat vertical = getStructuringElement(MORPH_RECT, Size(1, 15));

    // 开操作进行背景去噪
    morphologyEx(binary, morphImage, MORPH_OPEN, horizonal, Point(-1, -1), 2);
    morphologyEx(morphImage, morphImage, MORPH_OPEN, vertical, Point(-1, -1), 2);
    morphologyEx(morphImage, morphImage, MORPH_OPEN, smallKernel, Point(-1, -1), 2);

    // 闭操作进行联通物体内部
    morphologyEx(morphImage, morphImage, MORPH_CLOSE, bigKernel, Point(-1, -1), 2);
    imshow("morphology", morphImage);
    imwrite(output_dir + input_num + "/morph.jpg", morphImage);

    Mat paddedImage;
    copyMakeBorder(morphImage, paddedImage, 10, 10, 10, 10, BORDER_CONSTANT, 255);
    imshow("paddedImage", paddedImage);
    imwrite(output_dir + input_num + "/padded.jpg", paddedImage);

    // 获取最大轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hireachy;
    findContours(paddedImage, contours, hireachy, CV_RETR_LIST, CHAIN_APPROX_SIMPLE, Point());
    Mat connImage = Mat::zeros(src.size(), CV_8UC3);
    for (size_t t = 0; t < contours.size(); t++) {
        Rect rect = boundingRect(contours[t]);
        if( rect.width < src.cols / 2) continue;
        // if (rect.width > src.cols - 10) continue;

        double area = contourArea(contours[t]);
        double len = arcLength(contours[t], true);
       
        RotatedRect rrect = minAreaRect(contours[t]);
        Point2f vertex[4];
        rrect.points(vertex);

        //绘制旋转矩形
        for (int i = 0; i < 4; i++) {
            cv::line(connImage, vertex[i], vertex[(i + 1) % 4], cv::Scalar(255, 100, 200), 2, CV_AA);
        }

        drawContours(connImage, contours, t, Scalar(0, 0, 255), 1, 8, hireachy);
        printf("area of star could : %f \n", area);
        printf("lenght of star could : %f \n", len);
    }
    imshow("finalImage" + input_num, connImage);
    imwrite(output_dir + input_num + "/ContourWithRect.jpg", connImage);

    waitKey(0);
    return 0;
}