#include "getPosture.h"

/* 获得单个点经过旋转后所在精确坐标
* @param angle : 顺时针为正，逆时针为负 
*/
Point getRotatedPoint(Point inputPoint,Point center,double angle){
    Point rotatedPoint;
    rotatedPoint.x = (inputPoint.x - center.x) * cos(-1 * angle) - (inputPoint.y - center.y) * sin(-1 * angle) + center.x;
    rotatedPoint.y = (inputPoint.x - center.x) * sin(-1 * angle) + (inputPoint.y - center.y) * cos(-1 * angle) + center.y;
    return rotatedPoint;
}

Posture getPosture(const vector<Point>& contour, Mat& srcImg, bool drawResult = false) {
    Posture pos;
    getOrientation(contour, pos, srcImg, drawResult);
    getSize(contour, pos, srcImg, drawResult);
    return pos;
}

void getOrientation(const vector<Point>& contour, 
                    Posture& posture, 
                    Mat& srcImg,
                    bool drawResult = false) {
    // Construct a buffer used by the pca analysis
    size_t pointCount = contour.size(); 
    Mat pca_data = Mat(pointCount, 2, CV_64FC1); // n rows * 2 cols(x, y)

    for(size_t i = 0;i < pointCount;++i) {
        pca_data.at<double>(i, 0) = contour[i].x;
        pca_data.at<double>(i, 1) = contour[i].y;
    }

    // Perform PCA
    cv::PCA pca_analysis(pca_data, Mat(), CV_PCA_DATA_AS_ROW);

    posture.center.x = pca_analysis.mean.at<double>(0, 0); 
    posture.center.y = pca_analysis.mean.at<double>(0, 1); 
    
    // 2 eigenvectors/eigenvalues are enough
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);

    for (size_t i = 0; i < 2; ++i) {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i,0);
    }

    // Get the angle
    posture.angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x);

    if(drawResult) {
        // Draw the principal components
        // 在轮廓中点绘制小圆
        circle(srcImg, posture.center, 3, CV_RGB(255, 0, 255), 2);
        //计算出直线，在主要方向上绘制直线
        line(srcImg, posture.center, posture.center + 0.02 * Point2f(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]) , CV_RGB(255, 255, 0));
        line(srcImg, posture.center, posture.center + 0.02 * Point2f(eigen_vecs[1].x * eigen_val[1], eigen_vecs[1].y * eigen_val[1]) , CV_RGB(0, 255, 255));
    }
}

void getSize(const vector<Point>& contour, 
                    Posture& posture, 
                    Mat& srcImg,
                    bool drawResult = false) {
    vector<Point> rotated_contour = contour;
    size_t pointCount = contour.size(); 

    // 获取旋转后的轮廓
    for(size_t i = 0;i < pointCount;++i) {
        rotated_contour[i] = getRotatedPoint(contour[i], posture.center, posture.angle);
    }

    // 轮廓最小外接矩形
    Rect rect = boundingRect(rotated_contour);

    posture.width = std::min(rect.width,rect.height);
    posture.height = std::max(rect.width,rect.height);

    if(drawResult) {
        RotatedRect rotated_rect((Point2f)rect.tl(),Point2f(rect.br().x,rect.tl().y),(Point2f)rect.br());
        Point2f vertexes[4];
        rotated_rect.points(vertexes);

        for (int i = 0; i < 4; i++) {
            vertexes[i] = getRotatedPoint(vertexes[i], posture.center, -1 * posture.angle);
        }

        //绘制旋转矩形
        for (int i = 0; i < 4; i++) {
            cv::line(srcImg, vertexes[i], vertexes[(i + 1) % 4], cv::Scalar(255, 100, 200), 2, CV_AA);
        }

    }
}