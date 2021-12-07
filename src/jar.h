#ifndef JAR_H
#define JAR_H
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include "utils.h"

using namespace std;
using namespace cv;

typedef struct {
    Point2d center;

    double angle;
    double angle_double;

    int width;
    int height;
} Posture;

using ContourPtr = std::shared_ptr<vector<Point> >;
using ImgPtr = std::shared_ptr<Mat>;
using ConstImgPtr = std::shared_ptr<const Mat>;
using Bounds = pair<int, int>;

class Jar {
    enum State {
        kNotInited = 0,
        kInited,
        kPostureGot,
        kObstructionFound,
        kFinished,
        kError
    };

public:

    Jar() = default;
    bool init(const string& imgName);
    bool init(const Mat& img);
    bool init(Mat&& img);

    Posture& getPosture();
    void getObstruction();
    void drawResult(const string& output);

private:
    // 预处理罐体图片（滤波、去除高光等）
    void preprocess();

    // 通过直方图获取灰度阈值
    int getGrayThreshold(const Mat& gray);

    // 获取罐体的原始轮廓
    ContourPtr getOriginalContour(const Mat& gray, bool showContour = false, cv::Scalar color = CV_RGB(200, 150, 200));

    // 平滑轮廓
    void smoothenContour(ContourPtr contour, int filterRadius = 5);

    // 获取罐体与x轴正向的粗略夹角，顺时针为正，[-90, 90]
    double getRoughOrientationByPCA();

    // 获取罐体与x轴正向的精确夹角，顺时针为正，[-90, 90]
    double getOrientation();

    // 有可能有标签与背景联通的情况，进行填补修正
    ContourPtr fixContour();

    // 获取罐体在旋转后的轮廓
    ContourPtr getRotatedContour(ContourPtr contour, double angle);

    void scanContour(const Mat& img, const std::vector<Point>& contour);

    // 扫描方向
    enum Direction {
        Left = 0,
        Right,
        Up,
        Down
    };

    // 分析统计得到的宽度数据
    int handleWidths();

    // 扫描查找障碍并框选
    void findAndMarkObstruction(Direction d = Left, int obstruction_threshold = 20);

private:
    State state_ = kNotInited;
    Point ImgCenter_;
    Posture posture_; 
    
    /* 原始图片及灰度图 */
    ConstImgPtr srcImg_; // 用于像素处理的图，内容保持不变
    ImgPtr paintImg_; // 用于展示结果的图
    ImgPtr grayImg_; // 预处理得到的灰度图
    int  grayThreshold_ = 43; // 灰度图核密度估计结果
    ContourPtr originalContour_; // 原始灰度图得到的轮廓

    /* Hough变换检测出的直线及角度 */
    std::vector<Vec4d> lines_;
    std::vector<double> angles_;
    ContourPtr fixedContour_; // 边界填补后的轮廓

    /* 旋转相关的类成员 */
    ImgPtr rotatedGrayImg_; // 旋转后的灰度图像，用于扫描
    ContourPtr rotatedContour_; // 旋转为正的轮廓（中心点需要修正），用于辅助扫描
    Point2d rotatedCenter_; // 旋转并修正后的的轮廓中心点
    pair<int, int> verticalBounds_; // 旋转后的上下边界点
    int delta_col_ = 0; // 旋转后的坐标在x的偏移值
    int delta_row_ = 0; // 旋转后的坐标在y的偏移值

    /* 扫描记录 */
    std::unordered_map<int, Bounds> boundsOfY_; // key:Y轴坐标 val：罐体在该Y值下的左右边界;
    std::unordered_map<int, std::vector<Bounds> > allBoundsOfWidth_; // 每个宽度对应的Y值左右边界
    std::unordered_map<int, int> widthsCount_; // 每个宽度出现的次数
    
    /* 找到的障碍物 */
    std::vector<cv::RotatedRect> obstructions_;
};

#endif // JAR_H