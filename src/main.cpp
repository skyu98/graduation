#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include "jar.h"

using namespace std;
using namespace cv;


string input_dir = "../imgs/cv_imgs/";
string output_dir = "../output_imgs/";

int main(int argc, char* argv[]) {
    Jar jar;
  
    if(!jar.init(input_dir + "01.jpg"))
        return -1;
    jar.getPosture();
    jar.drawResult();
    return 0;
}