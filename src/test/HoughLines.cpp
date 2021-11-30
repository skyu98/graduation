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
    // imshow("0", channels[0]);
    // imshow("1", channels[1]);
    // imshow("2", channels[2]);

    Mat diff;
    absdiff(channels[2], channels[0], diff);

    my_utils::diff(channels[2], diff, channels[2], 2);

    Mat gray = std::move(channels[2]);

    // cv::GaussianBlur(gray, gray, Size(3, 3), 0, 0);

    Mat binary;
    // cv::adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 51, 5);
    // imshow("adaptive", binary);
    cout << cv::threshold(gray, binary, 45, 255, THRESH_BINARY) << endl;
    imshow("binary", binary);

    my_utils::morphology(binary, binary, MORPH_ERODE, 10);

    Mat paddedImg;
    cv::copyMakeBorder(binary, paddedImg, 1, 1, 1, 1, BORDER_CONSTANT, 255);

    // 获取轮廓
    vector<vector<Point> > contours;
    vector<Vec4i> hireachy;
    cv::findContours(paddedImg, contours, hireachy, CV_RETR_LIST, CHAIN_APPROX_SIMPLE, Point());

    Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    // 获取最大轮廓
    for (size_t t = 0; t < contours.size(); ++t) {
        Rect rect = boundingRect(contours[t]);
        if(rect.width < img.cols / 2) continue;

        drawContours(mask, contours, static_cast<int>(t), CV_RGB(255, 255, 255), 2, 8);
        // drawContours(img, contours, static_cast<int>(t), CV_RGB(255, 0, 0), 2, 8);
        break;
    } 
    
    vector<Vec4d> possible_lines;//定义一个存放直线信息的向量
						//Hough直线检测API
	cv::HoughLinesP(mask, possible_lines, 1, CV_PI / 180, 150, 200, 50);
	//InputArray src : 输入图像，必须8-bit的灰度图像；
    // OutputArray lines : 输出的极坐标来表示直线；
    // double rho : 生成极坐标时候的像素扫描步长；
    // double theta : 生成极坐标时候的角度步长，一般取值CV_PI/180；
    // int threshold : 阈值，只有获得足够交点的极坐标点才被看成是直线；
    // double minLineLength = 0 : 最小直线长度 
    // double maxLineGap = 0 : 最大间隔

    auto getAngle = [&](const Vec4d& line){
        Point A(line[0], line[1]);
        Point B(line[2], line[3]);
        if(line[0] > line[2]) {
           std::swap(A, B);
        }
        // cout << "(B.y - A.y): " << (B.y - A.y) << endl;
        // cout << "(B.x - A.x): " << (B.x - A.x) << endl;

        double k = static_cast<double>(B.y - A.y)/static_cast<double>(B.x - A.x);
        double line_arctan = static_cast<double>(atan(k));
        return line_arctan * 180.0 / M_PI;
    };

    //标记出直线
    int size = possible_lines.size();
    std::vector<double> angles;
    angles.reserve(size);
    double total = 0.0;
	for (size_t i = 0; i < size; i++) {
		const Vec4d& point1 = possible_lines[i];
		line(img, Point(point1[0], point1[1]), Point(point1[2], point1[3]), Scalar(255, 255, 0), 2, LINE_AA);
        
        double angle = getAngle(point1);
        angles.push_back(angle);
        total += angle;
	}

    double rough_average = total / static_cast<double>(angles.size());
    cout << rough_average << endl;

    total = 0.0;
    int count = 0;

    for(double angle : angles) {
        if(abs(angle - rough_average) < 20.0) {
            total += angle;
            ++count;
        }
    }
    double average_double = total / static_cast<double>(count);
    cout << average_double << endl;

    /* 旋转图片 */
    double average = (average_double / 180) * M_PI;
    Mat rotated_img = my_utils::rotateImage(img, 90 - average_double);

    int delta_col = (rotated_img.cols - img.cols) / 2;
    int delta_row = (rotated_img.rows - img.rows) / 2;

    Point center(img.cols / 2, img.rows / 2);
    int maxY = 0, minY = INT_MAX;
    
    for(int i = 0;i < size && (abs(angles[i] - average_double) < 10.0);++i) {
        Vec4d& line = possible_lines[i];
        Point A(line[0], line[1]);
        Point B(line[2], line[3]);
        
        A = my_utils::getRotatedPoint(A, center, M_PI_2 - average);
        B = my_utils::getRotatedPoint(B, center, M_PI_2 - average);
        A.x += delta_col;
        A.y += delta_row;

        B.x += delta_col;
        B.y += delta_row;

        maxY = max(maxY, A.y);
        maxY = max(maxY, B.y);

        minY = min(minY, A.y);
        minY = min(minY, B.y);
    }

    for(int i = 0;i < size && (abs(angles[i] - average_double) < 5.0);++i) {
        Vec4d& line = possible_lines[i];
        Point2d A(line[0], line[1]);
        Point2d B(line[2], line[3]);

        A = my_utils::getRotatedPoint(A, center, M_PI_2 - average);
        A.x += delta_col;
        A.y += delta_row;
        
        cv::line(rotated_img, A, Point2d(A.x, maxY), cv::Scalar(0, 0 ,0), 2, 4);
        cv::line(rotated_img, A, Point2d(A.x, minY), cv::Scalar(0, 0 ,0), 2, 4);

    }

    /* 旋转回去 */
    Mat tmp = my_utils::rotateImage(rotated_img, -1.0 * (90 - average_double));

    split(tmp, channels);
    // imshow("0", channels[0]);
    // imshow("1", channels[1]);
    // imshow("2", channels[2]);

    absdiff(channels[2], channels[0], diff);

    my_utils::diff(channels[2], diff, channels[2], 2);

    gray = std::move(channels[2]);

    cv::GaussianBlur(gray, gray, Size(3, 3), 0, 0);

    cv::threshold(gray, binary, 45, 255, THRESH_BINARY);

    my_utils::morphology(binary, binary, MORPH_ERODE, 15);

    cv::copyMakeBorder(binary, paddedImg, 1, 1, 1, 1, BORDER_CONSTANT, 255);

    // 获取轮廓
    vector<vector<Point> > contours2;
    vector<Vec4i> hireachy2;
    cv::findContours(paddedImg, contours2, hireachy2, CV_RETR_LIST, CHAIN_APPROX_SIMPLE, Point());

    // 获取最大轮廓
    for (size_t t = 0; t < contours2.size(); ++t) {
        Rect rect = boundingRect(contours2[t]);
        if(rect.width < img.cols / 2) continue;

        drawContours(tmp, contours2, static_cast<int>(t), CV_RGB(255, 0, 0), 2, 8);
        // drawContours(img, contours, static_cast<int>(t), CV_RGB(255, 0, 0), 2, 8);
        break;
    } 

    imshow("tmp", tmp);
    // imshow("rotated", rotated_img);
    imshow("mask", mask);
    imshow("img", img);
    waitKey(0);

    // Mat edges;
    return 0;
}