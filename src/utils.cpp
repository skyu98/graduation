#include "utils.h"

namespace my_utils {

/***********************************************************
增强算法的原理在于先统计每个灰度值在整个图像中所占的比例
然后以小于当前灰度值的所有灰度值在总像素中所占的比例，作为增益系数
对每一个像素点进行调整。由于每一个值的增益系数都是小于它的所有值所占
的比例和。所以就使得经过增强之后的图像亮的更亮，暗的更暗。
************************************************************/
void ImageStretchByHistogram(const Mat& src, Mat & dst) {
	//判断传入参数是否正常
	if (!(src.size().width == dst.size().width)) {
		cout << "error" << endl;
		return;
	}
	double p[256], p1[256], num[256];

	memset(p, 0, sizeof(p));
	memset(p1, 0, sizeof(p1));
	memset(num, 0, sizeof(num));
	int height = src.size().height;
	int width = src.size().width;
	long wMulh = height * width;

	//统计每一个灰度值在整个图像中所占个数
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			uchar v = src.at<uchar>(y, x);
			num[v]++;
		}
	}

	//使用上一步的统计结果计算每一个灰度值所占总像素的比例
	for (int i = 0; i < 256; i++)
	{
		p[i] = num[i] / wMulh;
	}

	//计算每一个灰度值，小于当前灰度值的所有灰度值在总像素中所占的比例
	//p1[i]=sum(p[j]);	j<=i;
	for (int i = 0; i < 256; i++)
	{
		for (int k = 0; k <= i; k++)
			p1[i] += p[k];
	}

	//以小于当前灰度值的所有灰度值在总像素中所占的比例，作为增益系数对每一个像素点进行调整。
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++) {
			uchar v = src.at<uchar>(y, x);
			dst.at<uchar>(y, x) = p1[v] * 255 + 0.5;
		}
	}
	return;
}

Point getRotatedPoint(Point inputPoint,Point center,double angle) {
    Point rotatedPoint;
    rotatedPoint.x = (inputPoint.x - center.x) * cos(-1 * angle) - (inputPoint.y - center.y) * sin(-1 * angle) + center.x;
    rotatedPoint.y = (inputPoint.x - center.x) * sin(-1 * angle) + (inputPoint.y - center.y) * cos(-1 * angle) + center.y;
    return rotatedPoint;
}

void putText(Mat& img, const string& text, Point textPos) {
	Scalar textColor(0,0,255);
	cv::putText(img, text.c_str(), textPos, CV_FONT_HERSHEY_SIMPLEX, 1, textColor, 2);
}


}