#include <opencv2/opencv.hpp>
#include <iostream>
#include <direct.h>
#include <math.h>
 
using namespace std;
using namespace cv;

Mat src, dst, gray_src;
char input_image[] = "input image";
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
    src = imread("../test_imgs/01.jpg");
    if (src.empty()) {
        printf("colud not load image ..\n");
        return -1;
    }

    namedWindow(input_image, CV_WINDOW_NORMAL);
    namedWindow(output_image, CV_WINDOW_NORMAL);
    // imshow(input_image, src);

    // 均值降噪
    Mat blurImg;
    GaussianBlur(src, blurImg, Size(11, 11), 0, 0);
    // imshow("blured", src);

    // 二值化
    Mat binary;
    cvtColor(blurImg, gray_src, COLOR_BGR2GRAY);
    threshold(gray_src, binary, 50, 200, THRESH_BINARY | THRESH_TRIANGLE);
    imshow("binary", binary);
    imwrite("../output_imgs/binary.jpg", binary);

    // 闭操作进行联通物体内部
    Mat morphImage;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, morphImage, MORPH_OPEN, kernel, Point(-1, -1), 5);
    morphologyEx(morphImage, morphImage, MORPH_CLOSE, kernel, Point(-1, -1), 2);
    imshow("morphology", morphImage);
    imwrite("../output_imgs/morph.jpg", morphImage);

    // 获取最大轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hireachy;
    findContours(morphImage, contours, hireachy, CV_RETR_LIST, CHAIN_APPROX_SIMPLE, Point());
    Mat connImage = Mat::zeros(src.size(), CV_8UC3);
    cout << contours.size() << endl;
    for (size_t t = 0; t < contours.size(); t++) {
        Rect rect = boundingRect(contours[t]);
        if (rect.width < src.cols / 2) continue;
        if (rect.width > src.cols - 20) continue;

        double area = contourArea(contours[t]);
        double len = arcLength(contours[t], true);


        drawContours(connImage, contours, t, Scalar(0, 0, 255), 1, 8, hireachy);
        printf("area of star could : %f \n", area);
        printf("lenght of star could : %f \n", len);
    }
    imshow(output_image, connImage);



    waitKey(0);
    return 0;
}