#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include "jar.h"
#include "imageCropper.h"
#include <chrono>
#include <unistd.h>

using namespace std;
using namespace cv;

string input_dir;
string output_dir;

imgCropper cropper(torch::kCPU);

FILE* file = nullptr;

void handleSingleJar(const string& imgName) {
    Mat img = cv::imread(input_dir + imgName);
    if(img.empty()) {
        printf("%s does not exist!!!\nPlease check the path...\n", imgName.c_str());
        return ;
    }
    cout << imgName << ":" << endl;

    cv::Rect box = cropper.getCroppedBox(img, 20);
    if(box.empty()) {
        cout << "No Jar Found in this img!" << endl;
        return;
    }

    // If you want to have an independent copy of the sub-array, use Mat::clone()
    Mat cropped = img(box).clone();
    // Mat padded;
    // cv::copyMakeBorder(cropped, padded, 10, 10, 10, 10, BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    Jar jar(imgName, box.tl());
    if(!jar.init(std::move(cropped))) return;

    jar.getPosture();
    jar.getObstruction();

    jar.drawResult(output_dir + imgName, false);
    jar.writResult(file);
    cout << "\n" << endl;
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        cout << "usage: ./traverseAll <InputDir> <OutputDir>(optional) " << endl;
        cout << "if <OutputDir> isn't given, an dir will be created in <InputDir>. " << endl;
        return 0;
    }

    input_dir = argv[1];
    if(input_dir.back() != '/') input_dir.push_back('/');
    if((access(input_dir.c_str(), 0)) == -1) {
        cout << "Input dir does not exist!" << endl;
        return 0;
    }

    output_dir = argc > 2 ? argv[2] : input_dir + "/output/";
    if((access(output_dir.c_str(), 0)) == -1) {
        mkdir(output_dir.c_str(), 0777);
        chmod(output_dir.c_str(), 07777);
    }

    auto start = std::chrono::high_resolution_clock::now();

    vector<string> imgNames;
    my_utils::traverseFolder(input_dir.c_str(), imgNames);
    cropper.init( "../yolo/yolov3.cfg", "../yolo/weights/jar.weights");

    // 先清空文件内容
    file = fopen((output_dir + "result.txt").c_str(), "w");
    fclose(file);

    file = fopen((output_dir + "result.txt").c_str(), "a+");
    for(const string& imgName: imgNames) {
        handleSingleJar(imgName);
    }
    fclose(file);

    auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> timeUsed = end - start;
	cout <<"Handled " << imgNames.size() << " jars in : " << timeUsed.count() << " ms" << endl;
    cout <<"The Average time for a single jar is :" << timeUsed.count() / imgNames.size() << endl;
    return 0;
}