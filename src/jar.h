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

class Jar {
public:
    Jar() = default;
    bool init(const string& imgName);

    ContourPtr getOriginalContour();
    Posture& getPosture();
    ContourPtr getRotatedContour();

    void findObstruction();
    void drawResult(const string& output);

private:
    // 获取罐体与x轴正向的夹角，顺时针为正，[-90, 90]
    void getOrientation();
    void getSize();

    bool inited = false;
    ImgPtr srcImg; // 用于像素处理的图，内容保持不变
    ImgPtr paintImg; // 用于展示结果的图
    ImgPtr rotatedImg; // 旋转后图像，罐体为直立状态
    Posture posture;
    ContourPtr originalContour;
    ContourPtr rotatedContour;
};

#endif // JAR_H