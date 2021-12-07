#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include "jar.h"
#include "imgCropper.h"
#include <Python.h>
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
    // imshow("origin", img);
    // waitKey(0);

    auto start = std::chrono::high_resolution_clock::now();

    imgCropper cropper;
    cropper.init("../yolo");
    cropper.findModule("detect");
    cropper.findFunc("findBox");
    cv::Rect box = cropper.getCroppedBox(img);
    if(box.empty()) {
        cout << "No Jar Found in this img!" << endl;
        return -1;
    }

    auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> timeUsed = end - start;
	cout << "YOLOV3 Time used : " << timeUsed.count() << " ms" << endl;

    Mat cropped = img(box);

    Jar jar;
    if(!jar.init(std::move(cropped)))
        return -1;

    jar.getPosture();
    jar.getObstruction();

    end = std::chrono::high_resolution_clock::now();
	timeUsed = end - start;
	cout << "Time used : " << timeUsed.count() << " ms" << endl;

    jar.drawResult(output_dir + imgName);
    return 0;
}