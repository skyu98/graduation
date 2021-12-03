#include <opencv2/opencv.hpp>
#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "imgCropper.h"
#include "utils.h"
using namespace cv;
using namespace std;

string input_dir = "../imgs/input_imgs/";
string output_dir = "../imgs/output_imgs/";

PyObject* matToNdarray(Mat& mat,int NPY_TYPE = NPY_UBYTE) { // Mat转Ndarray
    // 判断是否是连续的MAT，如果是ROI则不连续，需要将内存进行拷贝
	if(!mat.isContinuous()) { 
        mat = mat.clone(); 
    }

	npy_intp dim_np[]{mat.rows, mat.cols, mat.channels()};
	PyObject* pythonValue = PyArray_SimpleNewFromData(sizeof(dim_np) / sizeof(npy_intp), dim_np, NPY_TYPE, 
                                                    static_cast<void*>(mat.data));

    return pythonValue;
}

int init() {
    // 初始化Python
    Py_Initialize();
    // 检查初始化是否成功
    if (!Py_IsInitialized()) {
        return -1;
    }

    import_array();
}

int main(int argc, char* argv[]) {
    string imgName = argc >=2 ? argv[1] : "origin.jpg";
    cv::Mat image = imread(input_dir + imgName);
    Mat frame;
    cvtColor(image, frame, CV_RGB2GRAY);
    // imshow("frame", frame);
    // waitKey(0);

  
    // init();

    // // 添加路径
    // PyRun_SimpleString("import sys");
    // PyRun_SimpleString("sys.path.append('./')"); 
    // PyRun_SimpleString("sys.path.append('../src/test')");
 
    // // 载入脚本
    // string scriptName = "threshold";
    // PyObject* pModule = PyImport_ImportModule(scriptName.c_str());
    // if(!pModule) {
    //     printf("Can't find [%s].py!", scriptName.c_str());
    //     return -1;
    // }

    // // 识别出函数
    // string funcName = "display";
    // PyObject* pFunc= PyObject_GetAttrString(pModule, funcName.c_str());
    // if(!pFunc) {
    //     printf("Can't find function [%s]!", funcName.c_str());
    //     return -1;
    // }
    // if(!PyCallable_Check(pFunc)) {
    //     printf("Can't call function [%s]!", funcName.c_str());
    //     return -1;
    // }

    // // 构造参数
    // PyObject* PyArray  = matToNdarray(frame);
    // PyObject* ArgArray = PyTuple_New(1); //同样定义大小与Python函数参数个数一致的PyTuple对象
    // PyTuple_SetItem(ArgArray, 0, PyArray); 
    // // PyArrayObject* py_return = (PyArrayObject*)PyEval_CallObject(pFunc, ArgArray);
    // PyObject* py_return = PyEval_CallObject(pFunc, ArgArray);

    // // /* 关键： 反馈回来的数据直接将矩阵数据指针指向ArrayObject的数据即可 */
    // // cv::Mat img(cv::Size(frame.size[1], frame.size[0]), CV_8UC3);
    // // img.data = (u_char*)py_return->data;            //直接将图像的数据指针指向numpy返回的数据
    
    // int res = 0;
    // PyArg_Parse(py_return, "i", &res);
    // cout << res << endl;

    // // 关闭Python
    // Py_Finalize();
    imgCropper cropper;
    cropper.init("../yolo");
    cropper.findModule("detect");
    cropper.findFunc("findBox");
    auto res = cropper.getCroppedBox(image);
    // cout << res.first.x << endl; 
    // cout << res.second.y << endl; 
    return 0;
}
