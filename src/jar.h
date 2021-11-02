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
} Posture;

using PosturePtr = std::shared_ptr<Posture>;
using ContourPtr = std::shared_ptr<vector<Point> >;
using ImgPtr = std::shared_ptr<Mat>;

class Jar {
public:
    Jar() = default;
    bool init(const string& imgName);
    ContourPtr getOriginalContour();
    PosturePtr getPosture();
    ContourPtr getRotatedContour();
    void drawResult();

private:
    void getOrientation();
    void getSize();

    bool inited = false;
    ImgPtr srcImg;
    PosturePtr posture;
    ContourPtr originalContour;
    ContourPtr rotatedContour;
};

#endif // JAR_H