#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include "findContour.h"
#include "getPosture.h"

using namespace std;
using namespace cv;

// 0 - black 255 - white
string input_dir = "../test_imgs/";
string output_dir = "../output_imgs/";

int main(int argc, char argv[]) {
    Mat srcImg;
    srcImg = imread(input_dir + ".jpg");
    if (srcImg.empty()) {
        printf("colud not load image ..\n");
        return -1;
    }

    vector<Point> contour;
    getBiggestContour(srcImg, contour);
    auto posture = getPosture(contour, srcImg, true);

    imshow("finalImage", srcImg);
    // imwrite(output_dir + input_num + "/ContourWithRect.jpg", connImage);
    waitKey(0);

    return 0;
}