#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include "jar.h"
#include "imageCropper.h"
#include <chrono>

using namespace std;
using namespace cv;

string input_dir = "../imgs/raw_imgs/";
string output_dir = "../imgs/output_imgs/";

int main(int argc, char* argv[]) {
    string imgName = argc >= 2 ? argv[1] : "origin.jpg";
    Mat img = cv::imread(input_dir + imgName);
    if(img.empty()) {
        printf("%s does not exist!!!\nPlease check the path...\n", imgName.c_str());
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    imgCropper cropper(torch::kCPU);
    cropper.init( "../yolo/yolov3.cfg", "../yolo/weights/jar.weights");
    cv::Rect box = cropper.getCroppedBox(img, 30);
    if(box.empty()) {
        cout << "No Jar Found in this img!" << endl;
        return -1;
    }

    auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> timeUsed = end - start;
	cout << "YOLOV3 Time used : " << timeUsed.count() << " ms" << endl;

    // If you want to have an independent copy of the sub-array, use Mat::clone()
    Mat cropped = img(box).clone();
    // Mat padded;
    // cv::copyMakeBorder(cropped, padded, 10, 10, 10, 10, BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    Jar jar;
    if(!jar.init(std::move(cropped)))
        return -1;

    jar.getPosture();
    jar.getObstruction();

    end = std::chrono::high_resolution_clock::now();
	timeUsed = end - start;
	cout << "Time used : " << timeUsed.count() << " ms" << endl;

    jar.drawResult(output_dir + imgName, true);
    return 0;
}