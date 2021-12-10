#ifndef IMAGECROPPER_H
#define IMAGECROPPER_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <torch/torch.h>

#include "Darknet.h"

using namespace std;
using namespace cv;

// FIXME: singleton
class imgCropper {
public:
    imgCropper(torch::DeviceType device_type) : device_(device_type) {}

    int init(const string& cfgPath, const string& weightsPath, int input_image_size = 416);
    cv::Rect getCroppedBox(const Mat& src, int gap = 20);

    ~imgCropper() {
        delete net_;
    }
private:
    static torch::NoGradGuard no_grad;

    torch::Device device_;
    Darknet* net_;

    Point topLeft_;
    Point bottomRight_;
};

#endif // IMAGECROPPER_H