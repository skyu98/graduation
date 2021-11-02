#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include "jar.h"
using namespace cv;
using namespace std;


int main() {
    Mat img = imread("../imgs/cv_imgs/01.jpg");
    int row = img.size().height, col = img.size().width;

    // 均值降噪
    Mat blurImg;
    cv::GaussianBlur(img, blurImg, Size(9, 9), 0, 0);

    Mat gray;
    cvtColor(blurImg, gray, COLOR_BGR2GRAY);

    Mat edges;
    Canny(gray, edges, 30, 60);
    imshow("edges", edges);

    vector<Point> contour;
    contour.reserve(50000);
    for(int x = 0;x < col;++x) {
        for(int y = 0;y < row;++y) {
            if(edges.at<uchar>(y, x) == 255) {
                contour.emplace_back(x, y);
            }
        }
    }

    size_t pointCount = contour.size(); 
    Mat pca_data = Mat(pointCount, 2, CV_64FC1); // n rows * 2 cols(x, y)

    for(size_t i = 0;i < pointCount;++i) {
        pca_data.at<double>(i, 0) = contour[i].x;
        pca_data.at<double>(i, 1) = contour[i].y;
    }

    // Perform PCA
    cv::PCA pca_analysis(pca_data, Mat(), CV_PCA_DATA_AS_ROW);

    PosturePtr posture = make_shared<Posture>();

    posture->center.x = pca_analysis.mean.at<double>(0, 0); 
    posture->center.y = pca_analysis.mean.at<double>(0, 1); 
    
    // 2 eigenvectors/eigenvalues are enough
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);

    for (size_t i = 0; i < 2; ++i) {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i,0);
    }

    // Get the angle, range: (-pi, pi]
    posture->angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x);
    posture->angle_double = 180 * (posture->angle) / M_PI;
    cout << posture->angle_double << endl;

    // Draw the principal components
    // 在轮廓中点绘制小圆
    circle(img, posture->center, 3, CV_RGB(255, 0, 255), 2);
    //计算出直线，在主要方向上绘制直线
    line(img, posture->center, posture->center + 0.005 * Point2f(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]) , CV_RGB(255, 255, 0));

    // Mat morph;
    // Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    // // 开操作
    // morphologyEx(edges, morph, MORPH_CLOSE, kernel);
    // // 闭操作
    // // morphologyEx(morph, morph, MORPH_CLOSE, kernel);
    // imshow("morph", morph);

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