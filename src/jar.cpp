#include "jar.h"

bool Jar::init(const string& imgName) {
    Mat src = imread(imgName);
    if(src.empty()) {
        printf(" %s does not exist!\nPlease check the path...\n", imgName.c_str());
        return false;
    }
    srcImg = make_shared<Mat>(src);

    Mat paint = srcImg->clone();
    paintImg = make_shared<Mat>(paint);

    inited = true;    
    return true;
}

ContourPtr Jar::getOriginalContour() {
    if(originalContour) return originalContour;

    assert(inited);
    // 均值降噪
    Mat blurImg;
    cv::GaussianBlur(*paintImg, blurImg, Size(9, 9), 0, 0);

    // 顺序：B G R 
    Mat channels[3];
    split(blurImg, channels);

    // R - B 获得高光区域掩膜
    Mat diff;
    absdiff(channels[2], channels[0], diff);
    my_utils::morphology(diff, diff, MORPH_DILATE, 10);
    imshow("mask", diff);
    // 使用掩膜去除高光
    my_utils::diff(channels[2], diff, channels[2], 2);

    // 灰度图
    Mat grayImg = std::move(channels[2]);
    imshow("gray", grayImg);
   
    // 二值化
    Mat binaryImg;
    // 0 - black 255 - white
    cv::threshold(grayImg, binaryImg, 40, 255, THRESH_BINARY);

    // 使用 ERODE 方式 让轮廓外扩一些
    my_utils::morphology(binaryImg, binaryImg, MORPH_ERODE, 10);
    imshow("binaryImg", binaryImg);

    // 图片各方向填充一个像素，避免边缘的线条被识别为轮廓
    Mat paddedImg;
    cv::copyMakeBorder(binaryImg, paddedImg, 1, 1, 1, 1, BORDER_CONSTANT, 255);
    // imshow("paddedImg", paddedImg);

    // 获取轮廓
    vector<vector<Point> > contours;
    vector<Vec4i> hireachy;
    cv::findContours(paddedImg, contours, hireachy, CV_RETR_LIST, CHAIN_APPROX_SIMPLE, Point());

    // 获取最大轮廓
    for (size_t t = 0; t < contours.size(); ++t) {
        Rect rect = boundingRect(contours[t]);
        if( rect.width < paintImg->cols / 2) continue;

        drawContours(*paintImg, contours, t, CV_RGB(255, 0, 0), 2, 8);
        originalContour = make_shared<vector<Point> >(std::move(contours[t]));
        break;
    } 
    return originalContour;
}

Posture& Jar::getPosture() {
    if(posture.isComplete) return posture;
    
    getOriginalContour();
    getOrientation();
    getSize();
    posture.isComplete = true;
    return posture;
}

ContourPtr Jar::getRotatedContour() {
    if(rotatedContour) return rotatedContour;
    getPosture();
    return rotatedContour;
}

void Jar::getOrientation() {
    // 获取最小外接矩形的角度
    RotatedRect rotated_rect = minAreaRect(*originalContour);
    double rorated_rect_angle = rotated_rect.angle;
    if(rotated_rect.size.width < rotated_rect.size.height) {
        rorated_rect_angle += 90;
    }
    // cout << "rotated_rect: " << rorated_rect_angle << endl;
    
    //绘制旋转矩形
    // Point2f vertexes[4];
    // rotated_rect.points(vertexes);
    // for (int i = 0; i < 4; i++) {
    //     cv::line(*paintImg, vertexes[i], vertexes[(i + 1) % 4], cv::Scalar(255, 100, 200), 2, CV_AA);
    // }

    // Construct a buffer used by the pca analysis
    size_t pointCount = originalContour->size(); 
    Mat pca_data = Mat(pointCount, 2, CV_64FC1); // n rows * 2 cols(x, y)

    for(size_t i = 0;i < pointCount;++i) {
        pca_data.at<double>(i, 0) = (*originalContour)[i].x;
        pca_data.at<double>(i, 1) = (*originalContour)[i].y;
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

    // Get the eigenvec angle, range: (-pi, pi]
    double eigenvec_angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x);
    double eigenvec_angle_double = 180 * (eigenvec_angle) / M_PI;
    if(eigenvec_angle_double > 90) {
        eigenvec_angle_double -= 180;
    }
    else if(eigenvec_angle_double < -1 * 90) {
        eigenvec_angle_double += 180;
    }

    // 实际角度为两个角度的平均值
    // posture.angle_double = (eigenvec_angle_double + rorated_rect_angle) / 2;
    posture.angle_double = eigenvec_angle_double; 
    posture.angle = M_PI * posture.angle_double / 180;
    // cout << posture.angle_double << ", " << posture.angle << endl;

    // Draw the principal components
    // 在轮廓中点绘制小圆
    circle(*paintImg, posture.center, 3, CV_RGB(255, 0, 255), 2);
    //计算出直线，在主要方向上绘制直线
    line(*paintImg, posture.center, posture.center + 800 * Point2f(cos(posture.angle), sin(posture.angle)) , CV_RGB(255, 255, 0));
}

void Jar::getSize() {
    size_t pointCount = originalContour->size(); 

    Point center(paintImg->size().width / 2, paintImg->size().height / 2);
    // 获取旋转后的轮廓
    if(!rotatedContour) {
        rotatedContour = make_shared<vector<Point> >(pointCount);
        for(size_t i = 0;i < pointCount;++i) {
            (*rotatedContour)[i] = my_utils::getRotatedPoint((*originalContour)[i], center, posture.angle);
        }
    }
    
    // 轮廓最小外接矩形
    Rect rect = boundingRect(*rotatedContour);

    posture.width = std::min(rect.width,rect.height);
    posture.height = std::max(rect.width,rect.height);

    RotatedRect rotated_rect((Point2f)rect.tl(),Point2f(rect.br().x,rect.tl().y),(Point2f)rect.br());
    Point2f vertexes[4];
    rotated_rect.points(vertexes);

    for (int i = 0; i < 4; i++) {
        // vertexes[i] = my_utils::getRotatedPoint(vertexes[i], posture.center, -1 * posture.angle);
        vertexes[i] = my_utils::getRotatedPoint(vertexes[i], center, -1 * posture.angle);
    }

    //绘制旋转矩形
    for (int i = 0; i < 4; i++) {
        cv::line(*paintImg, vertexes[i], vertexes[(i + 1) % 4], cv::Scalar(255, 200, 100), 2, CV_AA);
    }
}

void Jar::findObstruction() {
    if(!rotatedImg) {
        rotatedImg = make_shared<Mat>(my_utils::rotateImage(*srcImg, 90 - posture.angle_double));
    }
    imshow("dst", *rotatedImg);
    waitKey(0);
}

void Jar::drawResult(const string& output) {
    string info = "width :" + to_string(static_cast<int>(posture.width)) + "  "
                + "height :" + to_string(static_cast<int>(posture.height)) + "  "
                + "angle :" + to_string(posture.angle_double); 

    my_utils::putText(*paintImg, info, posture.center);
    imshow("Result", *paintImg);
    waitKey(0);
    imwrite(output, *paintImg);
}

