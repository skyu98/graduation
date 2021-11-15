#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include "jar.h"

const int LEFT = 0;
const int RIGHT = 1;

using namespace cv;
using namespace std;

string input_dir = "../imgs/input_imgs/";
string output_dir = "../imgs/output_imgs/";

// Mat removeHighlight(const Mat& src) {
//     // 均值降噪
//     Mat img;
//     cv::GaussianBlur(src, img, Size(11, 11), 0, 0);

//     Mat gray;
//     // 通道分割方式
//     Mat channels[3];
//     split(img, channels);

//     Mat diff;
//     absdiff(channels[2], channels[0], diff);
//     my_utils::diff(channels[2], diff, channels[2], 2);
//     gray = std::move(channels[2]);

//     // 二值化
//     Mat binaryImg;
//     // 0 - black 255 - white
//     cv::threshold(gray, binaryImg, 48, 255, THRESH_BINARY);
//     return gray;
// }

Point scanWidth(const Mat& img, Point center, int gray_threshold, int width_threshold, int direction = RIGHT) {
    int rows = img.size().height, cols = img.size().width;
    assert(center.x <= cols & center.y <= rows);

    int curX = center.x, curY = center.y, last_bound = curX;

    const int SCAN_FRONT = 0; 
    const int SCAN_BACK = 1;

    int status = SCAN_FRONT; // 初始状态为 扫描前景

    int front_count = 0, back_count = 0;

    int flag = direction == RIGHT ? 1 : -1;

    // while(curX <= cols && curX >= 0) {
    //     // 遇到的点为黑色
    //     if(static_cast<int>(img.at<Vec3b>(curY, curX)[0]) <= gray_threshold) {
    //         // 如果在扫描背景时遇到黑色点，则开始计数黑色点，超过阈值则重新认定为扫描前景状态
    //         // （即之前认定的背景白色为干扰块）
    //         if((status == SCAN_BACK) & (++front_count > width_threshold)) {
    //             status = SCAN_FRONT;
    //             front_count = 0;
    //             back_count = 0;
    //         }
    //     }
    //     else { // 遇到白色像素点
    //         // 为分界点
    //         if(status == SCAN_FRONT) { 
    //             status = SCAN_BACK;
    //             last_bound = curX;
    //             front_count = 0;
    //         }
    //         else if(++back_count > width_threshold) {
    //             break;
    //         }
    //     }
    //     curX += flag;
    // }

    while(curX <= cols && curX >= 0) {
        // 如果当前在扫描前景
        if(status == SCAN_FRONT) {
            // 遇到的点为黑色
            if(static_cast<int>(img.at<Vec3b>(curY, curX)[0]) <= gray_threshold) {
                // 继续扫描
            }
            else { // 遇到的点为白色
                status = SCAN_BACK;
                last_bound = curX - flag;
                front_count = 0;
            }
        }   
        else { // 如果当前在扫描背景
            // 遇到的点为黑色
            if(static_cast<int>(img.at<Vec3b>(curY, curX)[0]) <= gray_threshold) {
                if(++front_count > (width_threshold / 2)) {
                    status = SCAN_FRONT;
                    back_count = 0;
                }
            }
            else {
                ++back_count;
                if(back_count > front_count) front_count = 0;
                if(back_count > width_threshold) break;
            }
        }
        curX += flag;
    }
    return {last_bound, curY};
}

int main(int argc, char* argv[]) {

    string imgName = argc >= 2 ? argv[1] : "1.jpg";
    Mat img = imread(input_dir + imgName);
    int rows = img.rows;

    for(int i = 150;i < rows;i += 10) {
        Point left = scanWidth(img, Point(1037, i), 48, 80, LEFT);
        Point right = scanWidth(img, Point(1037, i), 48, 80, RIGHT);
        cv::circle(img, left, 2, CV_RGB(255, 0, 0), 2);
        cv::circle(img, right, 2, CV_RGB(255, 0, 0), 2);
    }

    
    imshow("scan_res", img);
    waitKey(0);
    return 0;
}