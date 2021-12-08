#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <math.h>
#include "utils.h"

using namespace cv;
using namespace std;

void getKDE(const vector<int>& x_array, const vector<int>& data, double bandwidth, vector<double>& y) {
    auto gauss = [](double x)-> double{
        return (1.0 / sqrt(2.0 * M_PI)) * exp(-0.5 * (x * x));
    };

    int N = data.size();
    double tmp = static_cast<double>(N * bandwidth);
    for(int x : x_array){
        double res = 0;
        for(int i = 0;i < N;++i) {
            res += gauss((x - data[i]) / bandwidth);
        }
        res /= tmp;
        y.push_back(res);
    }
}

string input_dir = "../imgs/cropped_imgs/";
string output_dir = "../imgs/output_imgs/";

int main(int argc, char* argv[]) {
    // Main_Inpaint();
    string imgName = argc >= 2 ? argv[1] : "1.jpg";
    Mat img = imread(input_dir + imgName);
    int row = img.size().height, col = img.size().width;

    Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

    Mat channels[3];
    split(img, channels);
    // imshow("0", channels[0]);
    // imshow("1", channels[1]);
    // imshow("2", channels[2]);

    Mat diff;
    absdiff(channels[2], channels[0], diff);
    imshow("diff", diff);
    Mat binary;
    cv::threshold(diff, binary, 28, 255, THRESH_BINARY);

    Mat res;
    cv::bitwise_and(channels[0], binary, res);
    imshow("res", res);
    
    my_utils::diff(gray, res, gray, 1);
    imshow("gray", gray);

    cv::threshold(gray, binary, 60, 255, THRESH_BINARY);
    imshow("binary", binary);
    // waitKey(0);

    // Mat gray = channels[2];
    // resize(diff, gray, Size(row / 5, col / 5));

    // vector<int> data;
    // int g_cols = gray.cols, g_rows = gray.rows;
    // data.reserve(g_cols *  g_rows);
    // for(int x = 0;x < g_cols;++x) {
    //     for(int y = 0;y < g_rows;++y) {
    //         int val = static_cast<int>(gray.ptr<uchar>(y)[x]);
    //         data.push_back(val);
    //     }
    // }

    // vector<int> x_array;
    // x_array.reserve(256);
    // for(int i = 0;i < 256;++i) {
    //     x_array.push_back(i);
    // }

    // vector<double> y;
    // y.reserve(x_array.size());
    // getKDE(x_array, data, 5, y);

    // int size = y.size();
    // for(int i = 1;i < size - 1;++i) {
    //     if(y[i] < y[i - 1] && y[i] < y[i + 1]) {
    //         cout << i << endl;
    //         break;
    //     }
    // }
    waitKey(0);
} 