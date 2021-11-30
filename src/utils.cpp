#include "utils.h"

namespace my_utils {

// https://blog.csdn.net/wsp_1138886114/article/details/118694938
Mat rotateImage(const Mat& image, double angle, int flag) {
	Mat dst, M;
	int h = image.rows;
	int w = image.cols;
    
    Point center(image.size().width / 2, image.size().height / 2);
    M = getRotationMatrix2D(center, -1 * angle, 1.0);
    if(flag == FULL_IMG) {
        double cos = abs(M.at<double>(0, 0));
        double sin = abs(M.at<double>(0, 1));

        int new_w = cos * w + sin * h;
        int new_h = cos * h + sin * w;
        M.at<double>(0, 2) += (new_w / 2.0 - w / 2);
        M.at<double>(1, 2) += (new_h / 2.0 - h / 2);
        
        warpAffine(image, dst, M, Size(new_w, new_h), INTER_LINEAR, 0, Scalar(255, 255, 255));
    }
    else {
	    warpAffine(image, dst, M, image.size());
    }
	return dst;
}

void splitStrToVec(const string& str, const string& split, vector<string>& vec) {
    size_t beg = 0, end = str.find(split);
    size_t len = split.size();
    while (end != string::npos) {
        vec.emplace_back(str.substr(beg, end - beg));
        beg = end + len;
        end = str.find(split, beg);
    }  
    if(beg != str.size()) {
        vec.emplace_back(str.substr(beg));
    }
}

Point getRotatedPoint(Point inputPoint, Point center, double angle) {
    Point rotatedPoint;
    rotatedPoint.x = (inputPoint.x - center.x) * cos(angle) - (inputPoint.y - center.y) * sin(angle) + center.x;
    rotatedPoint.y = (inputPoint.x - center.x) * sin(angle) + (inputPoint.y - center.y) * cos(angle) + center.y;
    return rotatedPoint;
}

void putText(Mat& img, const string& text, Point textPos) {
	Scalar textColor(0,0,255);
	cv::putText(img, text.c_str(), textPos, CV_FONT_HERSHEY_SIMPLEX, 1, textColor, 2);
}

void morphology(Mat &imgIn, Mat &imgOut, int morpOp,
                int minThickess, int shape, int iter) {
    int size = minThickess / 2;
    Point anchor = Point(size, size);
    Mat element = getStructuringElement(shape, Size(2 * size + 1, 2 * size + 1), anchor);
    morphologyEx(imgIn, imgOut, morpOp, element, anchor, iter);
}

void diff(Mat& src, const Mat& mask, Mat& dst, int iterations) {
    if(iterations <= 0) return;
    for(int i = 0;i < iterations;++i) {
        cv::absdiff(src, mask, dst);
    } 
}

// https://www.freesion.com/article/3908799093/
void getKDE(const vector<int>& x_array, const vector<int>& data,  vector<double>& y, double bandwidth) {
    auto gauss = [](double x)-> double{
        return (1 / sqrt(2.0 * M_PI)) * exp(-0.5 * (x * x));
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

} // my_utils 
