#include "jar.h"

bool Jar::init(const string& imgName) {
    if(state > kNotInited) return true;

    Mat src = imread(imgName);
    if(src.empty()) {
        printf(" %s does not exist!\nPlease check the path...\n", imgName.c_str());
        return false;
    }
    Mat paint = src.clone();

    srcImg = make_shared<const Mat>(src);
    paintImg = make_shared<Mat>(paint);

    preprocess();
    state = kInited;    
    return true;
}

Posture& Jar::getPosture() {
    if(state >= kPostureGot) return posture;
    
    getContour();
    getOrientation();
    getSize();

    state = kPostureGot;
    return posture;
}


void Jar::findObstruction() {
    // 先进行姿态估计
    if(state < kPostureGot) {
        getPosture();
    }

    if(state >= kObstructionFound) return;

    rotatedGray = make_shared<Mat>(my_utils::rotateImage(*gray, 90 - posture.angle_double));
    // imwrite("../imgs/output_imgs/gray.jpg", *rotatedGray);

    // 修正旋转后的中心点
    Point rotatedCenter = my_utils::getRotatedPoint(posture.center, Point(srcImg->cols / 2, srcImg->rows / 2), M_PI_2 - posture.angle);
    Point center(rotatedCenter.x + ((rotatedGray->cols - srcImg->cols) / 2), 
                rotatedCenter.y + ((rotatedGray->rows - srcImg->rows) / 2));

    // 向上向下扫描，统计宽度并获得上下边界
    Point up_bound = scanVertically(*rotatedGray, center, UP);
    Point down_bound = scanVertically(*rotatedGray, center, DOWN);

    // 统计处理得到的宽度结果，返回其 主平均宽度
    int mainWidth = handleWidths();

    Point tmp = up_bound;
    while(tmp.y <= down_bound.y) {
        if(widths[tmp.y] > mainWidth + 15) {
            cv::circle(*rotatedGray, tmp, 3, CV_RGB(255, 255, 255), 2);
        }
        tmp.y += 10;
    }
    
    state = kFinished;

    imshow("dst", *rotatedGray);
    waitKey(0);
}

void Jar::drawResult(const string& output) {
    assert(state == kFinished);
    string info = "width :" + to_string(static_cast<int>(posture.width)) + "  "
                + "height :" + to_string(static_cast<int>(posture.height)) + "  "
                + "angle :" + to_string(posture.angle_double); 

    my_utils::putText(*paintImg, info, posture.center);
    imshow("Result", *paintImg);
    waitKey(0);
    imwrite(output, *paintImg);
}

void Jar::preprocess() {
    // 均值降噪
    Mat blurImg;
    cv::GaussianBlur(*srcImg, blurImg, Size(9, 9), 0, 0);

    // 顺序：B G R 
    Mat channels[3];
    cv::split(blurImg, channels);

    // R - B 获得高光区域掩膜
    Mat diff;
    cv::absdiff(channels[2], channels[0], diff);
    // 使用掩膜去除高光
    my_utils::diff(channels[2], diff, channels[2], 2);

    // 灰度图--CV_8UC1
    gray = make_shared<Mat>(std::move(channels[2]));
    // imshow("gray", *gray);
}

void Jar::getContour() {
    // 灰度图二值化
    Mat binaryImg;
    // 0 - black 255 - white
    cv::threshold(*gray, binaryImg, 45, 255, THRESH_BINARY);

    // 使用 ERODE 方式 让轮廓外扩一些
    my_utils::morphology(binaryImg, binaryImg, MORPH_ERODE, 8);
    // imshow("binaryImg", binaryImg);

    // 图片各方向填充一个像素，避免边缘的线条被识别为轮廓
    Mat paddedImg;
    cv::copyMakeBorder(binaryImg, paddedImg, 1, 1, 1, 1, BORDER_CONSTANT, 255);

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
    Point center(paintImg->cols / 2, paintImg->rows / 2);

    // 计算原始轮廓绕中心旋转后得到的点
    rotatedContour = make_shared<vector<Point> >(pointCount);
    for(size_t i = 0;i < pointCount;++i) {
        (*rotatedContour)[i] = my_utils::getRotatedPoint((*originalContour)[i], center, M_PI_2 - posture.angle);
    }
    
    // 轮廓最小外接矩形
    Rect rect = boundingRect(*rotatedContour);

    posture.width = std::min(rect.width,rect.height);
    posture.height = std::max(rect.width,rect.height);

    RotatedRect rotated_rect((Point2f)rect.tl(),Point2f(rect.br().x,rect.tl().y),(Point2f)rect.br());
    Point2f vertexes[4];
    rotated_rect.points(vertexes);

    for (int i = 0; i < 4; i++) {
        vertexes[i] = my_utils::getRotatedPoint(vertexes[i], center, posture.angle - M_PI_2);
    }

    //绘制旋转矩形
    for (int i = 0; i < 4; i++) {
        cv::line(*paintImg, vertexes[i], vertexes[(i + 1) % 4], cv::Scalar(255, 200, 100), 2, CV_AA);
    }
}

Point Jar::scanVertically(const Mat& img, Point center, int direction, int step) {
    assert(img.type() == CV_8UC1);

    Point left_bound, right_bound;
    int maxRows = img.rows;
    int flag = direction == UP ? -1 : 1;
    int curWidth = 0;

    while(center.y >= 0 && center.y <= maxRows) {
        center.y += flag * step;
        left_bound = scanHorizonally(img, center, LEFT);
        right_bound = scanHorizonally(img, center, RIGHT);

        curWidth = right_bound.x - left_bound.x;
        curWidth = (curWidth / 5) * 5;
        if(curWidth == 0) break;

        widths[center.y] = curWidth; 
        ++widthsCount[curWidth];

        cv::circle(*rotatedGray, left_bound, 2, CV_RGB(255, 255, 255), 1);
        cv::circle(*rotatedGray, right_bound, 2, CV_RGB(255, 255, 255), 1);
    }
    return center;
}

Point Jar::scanHorizonally(const Mat& img, Point center, int direction, int gray_threshold, int width_threshold) {
    int rows = img.rows, cols = img.cols;
    assert(center.x <= cols & center.y <= rows);

    int curX = center.x, curY = center.y, last_bound = curX;

    const int SCAN_FRONT = 0; 
    const int SCAN_BACK = 1;

    int status = SCAN_FRONT; // 初始状态为 扫描前景

    int front_count = 0, back_count = 0;

    int flag = direction == RIGHT ? 1 : -1;

    while(curX <= cols && curX >= 0) {
        // 如果当前在扫描前景
        if(status == SCAN_FRONT) {
            // 遇到的点为黑色
            if(static_cast<int>(img.at<uchar>(curY, curX)) <= gray_threshold) {
                // 继续扫描
            }
            else { // 遇到的点为白色
                status = SCAN_BACK;
                last_bound = curX - flag;
                front_count = 0;
            }
        }   
        else { // 如果当前在扫描背景
            // 遇到的点为黑色
            if(static_cast<int>(img.at<uchar>(curY, curX)) <= gray_threshold) {
                if(++front_count > (width_threshold / 3)) {
                    status = SCAN_FRONT;
                    back_count = 0;
                }
            }
            else {
                ++back_count;
                if(back_count > 3 * front_count) front_count = 0;
                if(back_count > width_threshold) break;
            }
        }
        curX += flag;
    }
    return {last_bound, curY};
}

int Jar::handleWidths() {
    typedef pair<int, int> WidthAndCount;

    struct Compare_WidthAndCount {
        bool operator()(const WidthAndCount& x, const WidthAndCount& y) const {
            if(x.second == y.second) return x.first > y.first;
            return x.second > y.second;
        }
    };  
    std::priority_queue<WidthAndCount, vector<WidthAndCount>, Compare_WidthAndCount> p_queue;

    auto it = widthsCount.begin();
    for(int i = 0;i < 2;++i, ++it) {
        p_queue.push(*it);
    }

    Compare_WidthAndCount cmp;
    while(it != widthsCount.end()) {
        if(cmp(*it, p_queue.top())) {
            p_queue.pop();
            p_queue.push(*it);
        } 
        ++it;
    }

    int mainWidth = 0, mainCount = 0;
    while(!p_queue.empty()) {
        auto it = p_queue.top();
        mainWidth += it.first * it.second;
        mainCount += it.second;
        p_queue.pop();
    }
    mainWidth /= mainCount;

    cout << "The main average width of the jar is :" << mainWidth << endl;
    return mainWidth;
}
