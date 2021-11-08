#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include "jar.h"

using namespace std;
using namespace cv;


string input_dir = "../imgs/input_imgs/";
string output_dir = "../imgs/output_imgs/";

int main(int argc, char* argv[]) {
    Jar jar;

    string imgName = argc >=2 ? argv[1] : "1.jpg";
    if(!jar.init(input_dir + imgName))
        return -1;
    jar.getPosture();
    jar.findObstruction();
    jar.drawResult(output_dir + imgName);
    return 0;
}