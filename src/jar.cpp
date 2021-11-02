#include "jar.h"

bool Jar::init(const string& imgName) {
    Mat src = imread(imgName);
    if(src.empty()) return false;

    srcImg = make_shared<Mat>(src);
    // src.convertTo(*srcImg, -1, 0.9, 5.0);   
    // imshow("srcImg", *srcImg);

    inited = true;    
    return true;
}

ContourPtr Jar::getOriginalContour() {
    if(originalContour) return originalContour;

    assert(inited);
    // 传统二值化办法：
    // 均值降噪
    Mat blurImg;
    cv::GaussianBlur(*srcImg, blurImg, Size(11, 11), 0, 0);
    
    // 灰度图
    Mat grayImg;
    cv::cvtColor(blurImg, grayImg, COLOR_BGR2GRAY);
   
    // 二值化
    Mat binaryImg;
    // 0 - black 255 - white
    cv::threshold(grayImg, binaryImg, 60, 255, THRESH_BINARY);

    // canny边缘检测

    Mat paddedImg;
    cv::copyMakeBorder(binaryImg, paddedImg, 1, 1, 1, 1, BORDER_CONSTANT, 255);
    imshow("paddedImg", paddedImg);


    // 获取轮廓
    Mat resImg = Mat::zeros(srcImg->size(), CV_8UC3);
    vector<vector<Point> > contours;
    vector<Vec4i> hireachy;
    cv::findContours(paddedImg, contours, hireachy, CV_RETR_LIST, CHAIN_APPROX_SIMPLE, Point());

    // 获取最大轮廓
    for (size_t t = 0; t < contours.size(); ++t) {
        Rect rect = boundingRect(contours[t]);
        if( rect.width < srcImg->cols / 2) continue;
        
        RotatedRect rotated_rect = minAreaRect(contours[t]);
        cout << "rotated_rect: " << rotated_rect.angle + 90 << endl;
        Point2f vertexes[4];
        rotated_rect.points(vertexes);

        //绘制旋转矩形
        for (int i = 0; i < 4; i++) {
            cv::line(*srcImg, vertexes[i], vertexes[(i + 1) % 4], cv::Scalar(255, 100, 200), 2, CV_AA);
        }

        drawContours(*srcImg, contours, t, CV_RGB(255, 0, 0), 2, 8);
        originalContour = make_shared<vector<Point> >(std::move(contours[t]));
        break;
    } 
    return originalContour;
}

PosturePtr Jar::getPosture() {
    if(posture) return posture;
    
    posture = make_shared<Posture>();
    getOriginalContour();
    getOrientation();
    getSize();
    return posture;
}

ContourPtr Jar::getRotatedContour() {
    if(rotatedContour) return rotatedContour;
    getPosture();
    return rotatedContour;
}

void Jar::getOrientation() {
    // Construct a buffer used by the pca analysis
    size_t pointCount = originalContour->size(); 
    Mat pca_data = Mat(pointCount, 2, CV_64FC1); // n rows * 2 cols(x, y)

    for(size_t i = 0;i < pointCount;++i) {
        pca_data.at<double>(i, 0) = (*originalContour)[i].x;
        pca_data.at<double>(i, 1) = (*originalContour)[i].y;
    }

    // Perform PCA
    cv::PCA pca_analysis(pca_data, Mat(), CV_PCA_DATA_AS_ROW);

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
    circle(*srcImg, posture->center, 3, CV_RGB(255, 0, 255), 2);
    //计算出直线，在主要方向上绘制直线
    line(*srcImg, posture->center, posture->center + 0.005 * Point2f(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]) , CV_RGB(255, 255, 0));
    // line(*srcImg, posture->center, posture->center + 0.008 * Point2f(eigen_vecs[1].x * eigen_val[1], eigen_vecs[1].y * eigen_val[1]) , CV_RGB(0, 255, 255));
}

void Jar::getSize() {
    size_t pointCount = originalContour->size(); 

    // 获取旋转后的轮廓
    if(!rotatedContour) {
        rotatedContour = make_shared<vector<Point> >(pointCount);
        for(size_t i = 0;i < pointCount;++i) {
            (*rotatedContour)[i] = my_utils::getRotatedPoint((*originalContour)[i], posture->center, posture->angle);
        }
    }
    
    // 轮廓最小外接矩形
    Rect rect = boundingRect(*rotatedContour);

    posture->width = std::min(rect.width,rect.height);
    posture->height = std::max(rect.width,rect.height);

    RotatedRect rotated_rect((Point2f)rect.tl(),Point2f(rect.br().x,rect.tl().y),(Point2f)rect.br());
    Point2f vertexes[4];
    rotated_rect.points(vertexes);

    for (int i = 0; i < 4; i++) {
        vertexes[i] = my_utils::getRotatedPoint(vertexes[i], posture->center, -1 * posture->angle);
    }

    //绘制旋转矩形
    for (int i = 0; i < 4; i++) {
        cv::line(*srcImg, vertexes[i], vertexes[(i + 1) % 4], cv::Scalar(255, 200, 100), 2, CV_AA);
    }
}

void Jar::drawResult() {
    string info = "width :" + to_string(static_cast<int>(posture->width)) + "  "
                + "height :" + to_string(static_cast<int>(posture->height)) + "  "
                + "angle :" + to_string(posture->angle_double); 

                
    my_utils::putText(*srcImg, info, posture->center);
    imshow("Result", *srcImg);
    waitKey(0);
}

