#include "jar.h"

bool Jar::init(const string& imgName) {
    if(state_ > kNotInited) return true;

    Mat src = imread(imgName);
    if(src.empty()) {
        printf("%s does not exist!!!\nPlease check the path...\n", imgName.c_str());
        return false;
    }
    cout << "Initializing....Please wait...." << endl;

    Mat paint = src.clone();

    srcImg_ = make_shared<const Mat>(src);
    paintImg_ = make_shared<Mat>(paint);

    state_ = kInited;    
    return true;
}

bool Jar::init(const Mat& img) {
    if(state_ > kNotInited) return true;

    Mat src = img.clone();
    if(src.empty()) {
        printf("Input Img is Empty!!!\nPlease check the img...\n");
        return false;
    }
    cout << "Initializing....Please wait...." << endl;

    Mat paint = src.clone();

    srcImg_ = make_shared<const Mat>(src);
    paintImg_ = make_shared<Mat>(paint);

    state_ = kInited;    
    return true;
}

bool Jar::init(Mat&& img) {
    if(state_ > kNotInited) return true;

    Mat src(std::move(img));
    img.data = nullptr;

    if(src.empty()) {
        printf("Input Img is Empty!!!\nPlease check the img...\n");
        return false;
    }
    cout << "Initializing....Please wait...." << endl;

    Mat paint = src.clone();

    srcImg_ = make_shared<const Mat>(src);
    paintImg_ = make_shared<Mat>(paint);

    state_ = kInited;    
    return true;
}

Posture& Jar::getPosture() {
    if(state_ >= kPostureGot) return posture_;
    cout << "Getting Posture...." << endl;

    preprocess();

    cout << "Getting Gray Threshold...." << endl;
    grayThreshold_ = getGrayThreshold(*grayImg_);
    cout << grayThreshold_ << endl;

    // 得到原始的轮廓
    cout << ">>> Getting Original Contour...." << endl;
    originalContour_ = getOriginalContour(*grayImg_, false);

    // 平滑轮廓，使得下一步效果更好
    smoothenContour(originalContour_, 8);

    // 根据轮廓进行Hough变换，得到罐体角度
    cout << ">>> Getting Orientation...." << endl;
    posture_.angle_double = getOrientation();;
    posture_.angle = M_PI * posture_.angle_double / 180;

    // 根据角度，获取旋转后的图片和数据
    rotatedGrayImg_ = make_shared<Mat>(my_utils::rotateImage(*grayImg_, 90 - posture_.angle_double));
    delta_col_ = (rotatedGrayImg_->cols - grayImg_->cols) / 2;
    delta_row_ = (rotatedGrayImg_->rows - grayImg_->rows) / 2;
    // imwrite("../imgs/output_imgs/grayImg_.jpg", *rotatedGrayImg_);
   
    // 根据得到的角度，画出边界竖线，使得边界上的标签不再联通   
    cout << ">>> Fixxing Original Contour...." << endl;
    fixedContour_ = fixContour();
    smoothenContour(fixedContour_, 8);

    // 根据罐体角度，将图片和轮廓旋转为正
    cout << ">>> Getting Rotated Contour...." << endl;
    rotatedContour_ = getRotatedContour(fixedContour_, posture_.angle);

    // 旋转为正后进行宽度扫描，可以得到罐体上下边界点
    cout << ">>> Scanning Contour...." << endl;
    scanContour(*rotatedGrayImg_, *rotatedContour_);

    // 统计扫描结果，可以得到主平均长度、罐体的旋转后中心、原始中心和上下边界点
    cout << ">>> Handling Widths...." << endl;
    posture_.width = handleWidths();
    if(posture_.width == -1) {
        state_ = kError;
        cout << "Error Occured!!" << endl;
        return posture_;
    }
    
    state_ = kPostureGot;
    return posture_;
}

void Jar::getObstruction() {
    // 先进行姿态估计
    if(state_ < kPostureGot) {
        getPosture();
    }

    if(state_ >= kObstructionFound) return;
    cout << "Finding Obstruction...." << endl;

    findAndMarkObstruction(Left, posture_.width / 25);
    findAndMarkObstruction(Right, posture_.width / 25);

    cout << "Finished!Found " << obstructions_.size() << " obstruction(s)!" << endl;

    for(auto& rotated_rect : obstructions_) {
        rotated_rect.center.x -= delta_col_;
        rotated_rect.center.y -= delta_row_;

        rotated_rect.center = my_utils::getRotatedPoint(rotated_rect.center, Point(srcImg_->cols / 2, srcImg_->rows / 2), posture_.angle - M_PI_2);
        rotated_rect.angle += static_cast<float>(posture_.angle_double - 90.0);
    }
    
    state_ = kObstructionFound;
}

void Jar::drawResult(const string& output) {
    if(state_ < kPostureGot) {
        cout << "Please get Posture/Obstruction first...." << endl;
        return;
    }
    cout << "Drawing Result...." << endl;
    
    // Draw the principal components
    // cv::circle(*rotatedGrayImg_, rotatedCenter_, 3, CV_RGB(255, 255, 255), 2);

    // 在轮廓中点绘制小圆
    cv::circle(*paintImg_, posture_.center, 3, CV_RGB(100, 200, 255), 2);

    // 在主要方向上绘制直线
    cv::line(*paintImg_, posture_.center, posture_.center + 800 * Point2d(cos(posture_.angle), sin(posture_.angle)) , CV_RGB(255, 125, 0), 2);
    
    // 绘制轮廓最小外接矩形
    Rect rect = boundingRect(*rotatedContour_);

    RotatedRect rotated_rect((Point2f)rect.tl(),Point2f(rect.br().x,rect.tl().y),(Point2f)rect.br());
    Point2f vertexes[4];
    rotated_rect.points(vertexes);

    for (int i = 0; i < 4; i++) {
        vertexes[i].x -= delta_col_;
        vertexes[i].y -= delta_row_;
        vertexes[i] = my_utils::getRotatedPoint(vertexes[i], ImgCenter_, posture_.angle - M_PI_2);
    }

    for (int i = 0; i < 4; i++) {
        cv::line(*paintImg_, vertexes[i], vertexes[(i + 1) % 4], CV_RGB(100, 200, 255), 2, CV_AA);
    } 

    // 打印罐体姿态信息
    std::vector<string> vec = {"width :" + to_string(static_cast<int>(posture_.width)), 
                            "height :" + to_string(static_cast<int>(posture_.height)),
                            "angle :" + to_string(static_cast<int>(posture_.angle_double))};

    int x = posture_.center.x, y = posture_.center.y, dy = 35;
    for(int i = 0;i < vec.size();++i) {
        my_utils::putText(*paintImg_, vec[i], Point(x, y + i * dy));
    }

    // 标注出障碍所在位置
    for(auto& rotated_rect : obstructions_) {
        Point2f vertexes[4];
        rotated_rect.points(vertexes);

        for (int i = 0; i < 4; i++) {
            cv::line(*paintImg_, vertexes[i], vertexes[(i + 1) % 4], cv::Scalar(50, 50, 255), 2, CV_AA);
        }
    }

    imshow("Result", *paintImg_);
    waitKey(0);
    imwrite(output, *paintImg_);
    state_ = kFinished;
}


/* 以下用户不可见 */
void Jar::preprocess() {
    // 顺序：B G R 
    Mat channels[3];
    cv::split(*srcImg_, channels);

    // R - B 获得高光区域掩膜
    Mat diff(srcImg_->size(), CV_8UC1);
    cv::absdiff(channels[2], channels[0], diff);

    // 使用掩膜去除高光
    my_utils::diff(channels[2], diff, channels[2], 2);

    // 灰度图--CV_8UC1
    grayImg_ = make_shared<Mat>(std::move(channels[2]));
    imshow("gray", *grayImg_);
    // cv::GaussianBlur(*grayImg_, *grayImg_, Size(3, 3), 0, 0);
    // imshow("grayImg_", *grayImg_);
}

// https://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-1d-clustering-method-in-scikit
int Jar::getGrayThreshold(const Mat& gray) {
    Mat cropped(Size(gray.rows / 5, gray.cols / 5), CV_8UC1);
    int c_cols = cropped.cols, c_rows = cropped.rows;
    resize(gray, cropped, Size(c_rows, c_cols));
    
    vector<int> data;
    data.reserve(c_cols *  c_rows);
    for(int x = 0;x < c_cols;++x) {
        for(int y = 0;y < c_rows;++y) {
            data.push_back(static_cast<int>(cropped.ptr<uchar>(x)[y]));
        }
    }

    vector<int> x_array(256, 0);
    x_array.reserve(256);
    for(int i = 0;i < 256;++i) {
        x_array[i]= i;
    }

    vector<double> y;
    y.reserve(x_array.size());
    my_utils::getKDE(x_array, data, y, 5);

    int size = y.size();
    double extreme_max = y[0], extreme_min = 0;
    int extreme_max_idx = 0, extreme_min_idx = 0;

    // 寻找极大值和极小值
    for(int i = 1;i < size - 1;++i) {
        if(y[i] > y[i - 1] && y[i] > y[i + 1]) {
            extreme_max_idx = i; 
            extreme_max = y[i];
        }
        if(y[i] < y[i - 1] && y[i] < y[i + 1]) {
            extreme_min_idx = i;
            extreme_min = y[i];
            break;
        }
    }
    // cout << extreme_max_idx << ", " << extreme_min_idx << endl;

    double mid = extreme_min + (extreme_max - extreme_min) * 0.05;
    double delta = mid / 15.0;

    int res = 43;
    for(int i = extreme_max_idx;i <= extreme_min_idx;++i) {
        if(abs(y[i] - mid) < delta) {
            res = i;
            break;
        }
    }
    res = max(res, 35);
    res = min(res, 50);
    return res;

}

ContourPtr Jar::getOriginalContour(const Mat& gray, bool showContour, cv::Scalar color) {
    // 灰度图二值化
    Mat binaryImg(srcImg_->size(), CV_8UC1);
    // 0 - black 255 - white
    cv::threshold(gray, binaryImg, grayThreshold_, 255, THRESH_BINARY);

    // // 使用 ERODE 方式 让目标外扩一些
    // my_utils::morphology(binaryImg, binaryImg, MORPH_DILATE, 5);
    my_utils::morphology(binaryImg, binaryImg, MORPH_ERODE, 15);
    
    // 图片各方向填充一个像素，避免边缘的线条被识别为轮廓
    Mat paddedImg;
    cv::copyMakeBorder(binaryImg, paddedImg, 1, 1, 1, 1, BORDER_CONSTANT, 255);

    // 获取轮廓
    vector<vector<Point> > contours;
    vector<Vec4i> hireachy;
    cv::findContours(paddedImg, contours, hireachy, cv::RETR_LIST, CHAIN_APPROX_SIMPLE, Point());

    // 获取最大轮廓
    for (size_t t = 0; t < contours.size(); ++t) {
        Rect rect = boundingRect(contours[t]);
        if(rect.width < paintImg_->cols / 2) continue;
        if(showContour) {
            drawContours(*paintImg_, contours, static_cast<int>(t), color, 2, 8);
        }
        return make_shared<vector<Point> >(std::move(contours[t]));
    } 
}

void Jar::smoothenContour(ContourPtr contour, int filterRadius) {
    // contour smoothing parameters for gaussian filter
    int filterSize = 2 * filterRadius + 1;

    size_t size = contour->size();
    size_t len = size + 2 * filterRadius;
    size_t idx = size - filterRadius;
    
    vector<float> x, y;
    for (size_t i = 0; i < len; i++) {
        x.push_back((*contour)[(idx + i) % size].x);
        y.push_back((*contour)[(idx + i) % size].y);
    }
    // filter 1-D signals
    vector<float> xFilt, yFilt;
    cv::medianBlur(x, xFilt, filterSize);
    cv::medianBlur(y, yFilt, filterSize);

    // build smoothed contour
    for (size_t i = filterRadius; i < size + filterRadius; i++) {
        (*contour)[i] = Point(xFilt[i], yFilt[i]);
    }
}

double Jar::getRoughOrientationByPCA() {
    size_t pointCount = originalContour_->size(); 
    Mat pca_data = Mat(static_cast<int>(pointCount), 2, CV_64FC1); // n rows * 2 cols(x, y)

    for(size_t i = 0;i < pointCount;++i) {
        pca_data.ptr<double>(i)[0] = (*originalContour_)[i].x;
        pca_data.ptr<double>(i)[1] = (*originalContour_)[i].y;
    }

    // Perform PCA
    cv::PCA pca_analysis(pca_data, Mat(), CV_PCA_DATA_AS_ROW);
    
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
    double eigenvec_angle_double = 180.0 * (eigenvec_angle) / M_PI;

    if(eigenvec_angle_double > 90) {
        eigenvec_angle_double -= 180;
    }
    else if(eigenvec_angle_double < -1 * 90) {
        eigenvec_angle_double += 180;
    }

    return eigenvec_angle_double;
}

double Jar::getOrientation() {
    Mat onlyContours = cv::Mat::zeros(srcImg_->size(), CV_8UC1);;
    vector<vector<Point> > contours{*originalContour_};
    drawContours(onlyContours, contours, 0, CV_RGB(255, 255, 255), 2, 8);

    double len = arcLength(*originalContour_, true);
    double minLineLength = len / 30.0;
    double maxLineGap = minLineLength / 3.0;
	// Hough直线检测API
    // InputArray src : 输入图像，必须8-bit的灰度图像；
    // OutputArray lines_ : 输出的极坐标来表示直线；
    // double rho : 生成极坐标时候的像素扫描步长；
    // double theta : 生成极坐标时候的角度步长，一般取值CV_PI/180；
    // int threshold : 阈值，只有获得足够交点的极坐标点才被看成是直线；
    // double minLineLength = 0 : 最小直线长度 
    // double maxLineGap = 0 : 最大间隔
    int iter_times = 0;
    while(iter_times < 5 && lines_.size() < 10) {
        lines_.clear();
        cv::HoughLinesP(onlyContours, lines_, 1, CV_PI / 180, 100, minLineLength, maxLineGap);
        minLineLength *= 0.9;
        maxLineGap *= 1.1;
        ++iter_times;
    }
    
    auto getAngle = [&](const Vec4d& line){
        Point2d A(line[0], line[1]);
        Point2d B(line[2], line[3]);
        if(line[0] > line[2]) {
           std::swap(A, B);
        }

        if(B.x - A.x == 0.0) return 90.0;

        double k = (B.y - A.y)/(B.x - A.x);
        double line_arctan = static_cast<double>(atan(k));
        return line_arctan * 180.0 / M_PI;
    };

    angles_.reserve(lines_.size());
    double total = 0.0;
	for (const auto& line : lines_) {
		// cv::line(*paintImg_, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(255, 255, 0), 2, LINE_AA);
        double angle = getAngle(line);
        angles_.push_back(angle);
        total += angle;
	}

    double rough = getRoughOrientationByPCA();
    total = 0.0;
    int count = 0;
    for(double angle : angles_) {
        if(abs(angle - rough) < 20.0) {
            total += angle;
            ++count;
        }
    }
    return total / static_cast<double>(count);
}

ContourPtr Jar::fixContour() {
    /* 旋转图片 */
    ImgCenter_ = Point(srcImg_->cols / 2, srcImg_->rows / 2);

    Mat tmp = grayImg_->clone();

    int size = lines_.size();
    int maxY = 0, minY = INT_MAX;
    int maxX = 0, minX = INT_MAX;
    
    for(int i = 0;i < size && (abs(angles_[i] - posture_.angle_double) < 10.0);++i) {
        Vec4d& line = lines_[i];
        Point A = my_utils::getRotatedPoint(Point((line[0] + line[2]) / 2, (line[1] + line[3]) / 2),
                                        ImgCenter_, M_PI_2 - posture_.angle);

        maxX = max(maxX, A.x);
        minX = min(minX, A.x);
    }

    for(const Point& p : *originalContour_) {
        Point rotated_point = my_utils::getRotatedPoint(p, ImgCenter_, M_PI_2 - posture_.angle);
        maxY = max(maxY, rotated_point.y);
        minY = min(minY, rotated_point.y);
    }
    maxY -= 80;
    minY += 80;

    int average_x = minX + ((maxX - minX) >> 1);
    int left_x = 0, right_x = 0;
    int left_count = 0, right_count = 0;

    for(int i = 0;i < size && (abs(angles_[i] - posture_.angle_double) < 10.0);++i) {
        Vec4d& line = lines_[i];
        // cv::line(*paintImg_, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(255, 255, 0), 2, LINE_AA);

        Point2d A((line[0] + line[2]) / 2.0, (line[1] + line[3]) / 2.0);
        Point rotated_A = my_utils::getRotatedPoint(A, ImgCenter_, M_PI_2 - posture_.angle);
    
        // left
        if(rotated_A.x < average_x && rotated_A.x - minX <= 100) { 
            left_x += rotated_A.x;
            ++left_count;
        }
        //right
        if(rotated_A.x > average_x && maxX - rotated_A.x <= 100) {
            right_x += rotated_A.x;
            ++right_count;
        } 
    }

    if(left_count > 0) {
        left_x /= left_count;
        left_x += 10;

        Point left_Point(left_x, minY + (maxY - minY) / 2);
        Point2d end_up = my_utils::getRotatedPoint(Point2d(left_x, maxY), ImgCenter_, posture_.angle - M_PI_2);
        Point2d end_down = my_utils::getRotatedPoint(Point2d(left_x, minY), ImgCenter_, posture_.angle - M_PI_2);
        left_Point = my_utils::getRotatedPoint(left_Point, ImgCenter_, posture_.angle - M_PI_2);

        cv::line(tmp, left_Point, end_up, cv::Scalar(0, 0 ,0), 2, 4);
        cv::line(tmp, left_Point, end_down, cv::Scalar(0, 0 ,0), 2, 4);
    }
    if(right_count > 0) {
        right_x /= right_count;
        right_x -= 10;

        Point right_Point(right_x, minY + (maxY - minY) / 2);
        Point2d end_up = my_utils::getRotatedPoint(Point2d(right_x, maxY), ImgCenter_, posture_.angle - M_PI_2);
        Point2d end_down = my_utils::getRotatedPoint(Point2d(right_x, minY), ImgCenter_, posture_.angle - M_PI_2);
        right_Point = my_utils::getRotatedPoint(right_Point, ImgCenter_, posture_.angle - M_PI_2);
        
        cv::line(tmp, right_Point, end_up, cv::Scalar(0, 0 ,0), 3, 4);
        cv::line(tmp, right_Point, end_down, cv::Scalar(0, 0 ,0), 3, 4);
    }

    // imshow("tmp", tmp);
    return getOriginalContour(tmp, true);
}

ContourPtr Jar::getRotatedContour(ContourPtr contour, double angle) {
    // 计算原始轮廓绕中心旋转后得到的点，并修正旋转后的轮廓
    size_t pointCount = contour->size(); 
    auto rotated = make_shared<vector<Point> >(pointCount);
    for(size_t i = 0;i < pointCount;++i) {
        Point& point = (*rotated)[i];
        point = my_utils::getRotatedPoint((*contour)[i], ImgCenter_, M_PI_2 - angle);
        point.x += delta_col_;
        point.y += delta_row_;
    }
    return rotated;
}

void Jar::scanContour(const Mat& img, const std::vector<Point>& contour) {
    Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    int rows = img.rows, cols = img.cols;
    for(int y = 0;y <= rows;y += 10) {
        line(mask, Point(0, y), Point(cols, y), Scalar(255,255,255), 1, 4);
    }
    
    Mat onlyContours = cv::Mat::zeros(img.size(), CV_8UC1);;
    vector<vector<Point> > contours{*rotatedContour_};
    drawContours(onlyContours, contours, 0, CV_RGB(255, 255, 255), 1, 8);

    Mat res(img.size(), CV_8UC1);
    cv::bitwise_and(mask, onlyContours, res);

    vector<Point> locations;
    locations.reserve(2000);
    cv::findNonZero(res, locations); 

    sort(locations.begin(), locations.end(), [](const Point& a, const Point& b) {
        if(a.y == b.y) return a.x < b.x;
        return a.y < b.y; 
    });

    size_t size = locations.size();

    verticalBounds_.first = locations[0].y;
    verticalBounds_.second = locations[size - 1].y;
    posture_.height = verticalBounds_.second - verticalBounds_.first;
    
    size_t i = 0;
    int curX = 0, curY = 0;

    while(i < size) {
        curY = locations[i].y;
        int minX = locations[i].x;
        while(i < size && locations[i].y == curY) {
            ++i;
        }
        int maxX = locations[i - 1].x;
      
        int width = (maxX - minX) / 2;
        width *= 2;

        ++widthsCount_[width];
        boundsOfY_[curY] = make_pair(minX, maxX);
        allBoundsOfWidth_[width].emplace_back(minX, maxX);
    }
    return;
}

int Jar::handleWidths() {
    if(widthsCount_.empty()) return -1;
    typedef pair<int, int> WidthAndCount;

    struct Compare_WidthAndCount {
        bool operator()(const WidthAndCount& x, const WidthAndCount& y) const {
            if(x.second == y.second) return x.first > y.first;
            return x.second > y.second;
        }
    };  
    std::priority_queue<WidthAndCount, vector<WidthAndCount>, Compare_WidthAndCount> p_queue;

    auto it = widthsCount_.begin();
    int widthChoosed = widthsCount_.size() > 10 ? 10 : widthsCount_.size();
    
    for(int i = 0;i < widthChoosed;++i, ++it) {
        p_queue.push(*it);
    }

    Compare_WidthAndCount cmp;
    while(it != widthsCount_.end()) {
        if(cmp(*it, p_queue.top())) {
            p_queue.pop();
            p_queue.push(*it);
        } 
        ++it;
    }

    auto averageOfBoundsVec = [&](const std::vector<Bounds>& vec)->int{
        if(vec.empty()) return 0;
        int average = 0;
        for(const Bounds& bounds : vec) {
            average += bounds.first + ((bounds.second - bounds.first) >> 1);
        }
        return average / vec.size();
    };

    int mainWidth = 0, mainCount = 0;
    int averageX = 0;
    while(!p_queue.empty()) {
        auto& pair = p_queue.top();
        int curWidth = pair.first, curCount = pair.second;

        // 统计主平均宽度
        mainWidth += curWidth * curCount;
        mainCount += curCount;

        // 计算中心点x坐标
        auto& vec = allBoundsOfWidth_[curWidth];
        averageX += averageOfBoundsVec(vec);

        p_queue.pop();
    }
    mainWidth /= mainCount;

    rotatedCenter_.x = averageX / widthChoosed;
    rotatedCenter_.y = verticalBounds_.first + ((verticalBounds_.second - verticalBounds_.first) >> 1);

    posture_.center = my_utils::getRotatedPoint(Point(static_cast<int>(rotatedCenter_.x) - delta_col_, static_cast<int>(rotatedCenter_.y) - delta_row_), 
                                           ImgCenter_, posture_.angle - M_PI_2);

    return mainWidth;
}

void Jar::findAndMarkObstruction(Direction direction, int obstruction_threshold) {
    int up_bound = verticalBounds_.first, down_bound = verticalBounds_.second;
    int mainWidth = posture_.width;

    auto isObstruction = [&](int curY)->bool{
        int width = boundsOfY_[curY].second - boundsOfY_[curY].first;
        if(direction == Left) {
            int left_width = static_cast<int>(rotatedCenter_.x) - boundsOfY_[curY].first;
            return (width >= mainWidth + obstruction_threshold) &&
                (left_width >= mainWidth / 2 + obstruction_threshold);
        }
        else {
            int right_width = boundsOfY_[curY].second - static_cast<int>(rotatedCenter_.x);  
            return (width >= mainWidth + obstruction_threshold) &&
                (right_width >= mainWidth / 2 + obstruction_threshold);
        }
        return false;
    };

    int step = 5;
    int curY = up_bound;

    int begin = curY, end = curY, tmp_end = curY;
    int bound = direction == Left ? INT_MAX : 0;
    bool lastStepIsObstruction = false;

    while(curY <= down_bound) {
        // 当前属于障碍区域
        if(isObstruction(curY)) {
            // 如果上一步不是在障碍区扫描，则认为找到了下一个begin
            if(!lastStepIsObstruction) {
                bound = direction == Left ? INT_MAX : 0;
                lastStepIsObstruction = true;
                begin = curY;
            }

            // 记录障碍区域的边界
            bound = direction == Left ? min(boundsOfY_[curY].first, bound) : max(boundsOfY_[curY].second, bound);

            // cv::circle(*rotatedGrayImg_, Point(bound, curY), 3, CV_RGB(255, 255, 255), 2);
        }
        else { // 当前不是障碍区域
            // 如果上一步正在障碍区扫描，且下一步仍然不是障碍区域，则认为找到了一个end
            bool nextStepIsObstruction = isObstruction(curY + step);
            if(lastStepIsObstruction && !nextStepIsObstruction) {
                end = curY;
                // 忽略高度小于阈值的障碍
                if(end - begin >= obstruction_threshold) {
                    // 框出障碍
                    int width = static_cast<int>(abs(rotatedCenter_.x - bound)) - mainWidth / 2;
                    int height = end - begin;
                    Point2d center(0, begin + height / 2);
                    center.x = direction == Left ? (bound + width / 2): (bound - width / 2);
                    
                    // 忽略宽度小于阈值的障碍
                    if(width >= obstruction_threshold) {
                        obstructions_.emplace_back(center, Size(width, height), 0);
                    }
                }
                lastStepIsObstruction = false;
            }
        }
        curY += step;
    }
}

