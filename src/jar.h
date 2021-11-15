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
    Point2f center;

    double angle;
    double angle_double;

    double width;
    double height; 
    bool isComplete = false;
} Posture;

using ContourPtr = std::shared_ptr<vector<Point> >;
using ImgPtr = std::shared_ptr<Mat>;
using ConstImgPtr = std::shared_ptr<const Mat>;

class Jar {
    enum State {
        kNotInited = 0,
        kInited,
        kPostureGot,
        kObstructionFound,
        kFinished
    };

public:
    Jar() = default;
    bool init(const string& imgName);
    
    Posture& getPosture();
    void findObstruction();
    void drawResult(const string& output);

private:
    // 预处理罐体图片（滤波、去除高光等）
    void preprocess();

    // 获取罐体的轮廓
    void getContour();

    // 获取罐体与x轴正向的夹角，顺时针为正，[-90, 90]
    void getOrientation();

    // 获取罐体的尺寸
    void getSize();
    
    // 扫描方向
    enum direction {
        LEFT = 0,
        RIGHT,
        UP,
        DOWN
    };

    /** @brief 横向扫描得到边界点
     @param img 被扫描的图片，需要是灰度图（8UC1) 
     @param center 扫描的起点，物体中心 
     @param direction 扫描的方向，LEFT 或 RIGHT
     @param gray_threshold 灰度阈值，大于该值则可能为背景
     @param width_threshold 宽度阈值，连续扫描到的点大于该值则认定为背景
     @return 水平方向扫描到的边界点
    */
    Point scanHorizonally(const Mat& img, Point center, int direction = RIGHT, int gray_threshold = 45, int width_threshold = 80);

    /** @brief 从中心开始竖向移动，并对两边横向扫描得到左右两个边界点；结果保存在两个map中
        @param img 被扫描的图片，需要是灰度图（8UC1) 
        @param center 扫描的起点，物体中心 
        @param direction 扫描的方向，LEFT 或 RIGHT
        @param gray_threshold 灰度阈值，大于该值则可能为背景
        @param width_threshold 宽度阈值，连续扫描到的点大于该值则认定为背景
        @return 竖直方向扫描到的边界点
    */
    Point scanVertically(const Mat& img, Point center, int direction = UP, int step = 4);

    int handleWidths();

    State state = kNotInited;
    ConstImgPtr srcImg; // 用于像素处理的图，内容保持不变
    ImgPtr paintImg; // 用于展示结果的图

    ImgPtr gray; // 预处理得到的灰度图
    ContourPtr originalContour; // 原始灰度图得到的轮廓

    ImgPtr rotatedGray; // 旋转后的灰度图像，用于扫描
    ContourPtr rotatedContour; // 旋转为正的轮廓（中心点需要修正），用于辅助扫描

    Posture posture; 

    std::unordered_map<int, int> widths;  // key:Y轴坐标 val：该Y值下的罐体宽度
    std::unordered_map<int, int> widthsCount; // 每个宽度出现的次数（按5取整）
};

#endif // JAR_H