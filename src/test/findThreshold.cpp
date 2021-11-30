#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include "utils.h"

using namespace cv;
using namespace std;


string input_dir = "../imgs/input_imgs/";
string output_dir = "../imgs/output_imgs/";

int main(int argc, char* argv[]) {
    // Main_Inpaint();
    string imgName = argc >= 2 ? argv[1] : "1.jpg";
    Mat img = imread(input_dir + imgName);
    int rows = img.rows, cols = img.cols;

    // 均值降噪
    Mat blurImg;
    cv::GaussianBlur(img, blurImg, Size(3, 3), 0, 0);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    int total = 0;
    cout << static_cast<int>(gray.at<uchar>(Point(0, 0))) << endl;
    for(int x = 0;x < cols;++x) {
        for(int y = 0;y < rows;++y) {
            total += static_cast<int>(gray.at<uchar>(Point(x, y)));
        }
    }
    cout << total << endl;
    total /= (rows * cols);
    cout << total << endl;
}
