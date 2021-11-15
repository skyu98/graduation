#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
bool flag = false;
int xvalue = 0;
int yvalue = 0;
Mat image, image1, image2;
void mousecallback(int event, int x, int y, int flags, void* userdata);


string input_dir = "../imgs/input_imgs/";
string output_dir = "../imgs/output_imgs/";

int main(int argc, char* argv[]) {
    string imgName = argc >= 2 ? argv[1] : "1.jpg";
	namedWindow("imageshow", 0);
	Mat image = imread(input_dir + imgName, 1);
	if (!image.data)
	{
		cout << "the image is error" << endl;
		return 0;
	}
	imshow("imageshow", image);
	image.copyTo(image1);
	cv::setMouseCallback("imageshow", mousecallback, 0);
	waitKey(0);
	return 0;
}

void mousecallback(int event, int x, int y, int flags, void* userdata)
{
	image1.copyTo(image2);
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
	{
		flag = true;
	}
	break;
	case EVENT_LBUTTONUP:
	{
		if (flag)
		{
			xvalue = x;
			yvalue = y;
			flag = 0;
			int b = image1.at<Vec3b>(yvalue, xvalue)[0];
			int g = image1.at<Vec3b>(yvalue, xvalue)[1];
			int r = image1.at<Vec3b>(yvalue, xvalue)[2];
			
			cout <<"X: "<< xvalue << " Y: "<< yvalue << " B:" << b << ' ' << "G:" << g << ' ' << "R:" << r << endl;
		}
	}
	break;
	}
}