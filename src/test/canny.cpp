#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include "jar.h"
using namespace cv;
using namespace std;

string input_dir = "../imgs/input_imgs/";
string output_dir = "../imgs/output_imgs/";

int main(int argc, char* argv[]) {

    string imgName = argc >=2 ? argv[1] : "1.jpg";
    Mat img = imread(input_dir + imgName);
    int row = img.size().height, col = img.size().width;

    // 均值降噪
    Mat blurImg;
    cv::GaussianBlur(img, blurImg, Size(11, 11), 0, 0);

    Mat gray;
    cvtColor(blurImg, gray, COLOR_BGR2GRAY);

    Mat edges;
    Canny(gray, edges, 40, 80, 3, true);
    imshow("edges", edges);

    Mat morph;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    // 闭操作
    morphologyEx(edges, morph, MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);
    // // 开操作
    // morphologyEx(morph, morph, MORPH_OPEN, kernel);
    
    imshow("morph", morph);

    // vector<vector<Point> > contours;
    // findContours(edges.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    // vector<int> indices(contours.size());
    // iota(indices.begin(), indices.end(), 0);

    // sort(indices.begin(), indices.end(), [&contours](int lhs, int rhs) {
    //     return contours[lhs].size() > contours[rhs].size();
    // });

    // int N = 2; // set number of largest contours
    // N = min(N, int(contours.size()));

    // Mat3b res = img.clone();

    // // Draw N largest contours
    // for (int i = 0; i < N; ++i)
    // {
    //     Scalar color(rand() & 255, rand() & 255, rand() & 255);
    //     Vec3b otherColor(color[2], color[0], color[1]);

    //     drawContours(res, contours, indices[i], color, CV_FILLED);

    //     // Create a mask for the contour
    //     Mat1b res_mask(img.rows, img.cols, uchar(0));
    //     drawContours(res_mask, contours, indices[i], Scalar(255), CV_FILLED);

    //     // AND with edges
    //     res_mask &= edges;

    //     // remove larger contours
    //     drawContours(res_mask, contours, indices[i], Scalar(0), 2);

    //     for (int r = 0; r < img.rows; ++r)
    //     {
    //         for (int c = 0; c < img.cols; ++c)
    //         {
    //             if (res_mask(r, c))
    //             {
    //                 res(r,c) = otherColor;
    //             }
    //         }
    //     }
    // }

    imshow("Image", img);
    // imshow("N largest contours", res);
    waitKey();

    return 0;
}