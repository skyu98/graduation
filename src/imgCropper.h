#ifndef IMGCROPPER_H
#define IMGCROPPER_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <Python.h>
#include <numpy/arrayobject.h>

using namespace std;
using namespace cv;

class imgCropper {
public:
    imgCropper(){}

    int init(const string& scriptPath = "./");
    int findModule(const string& scriptName);
    int findFunc(const string& funcName);
    cv::Rect getCroppedBox(const Mat& src);

    ~imgCropper();
private:
    string scriptPath_;

    PyObject* pyModule_;
    PyObject* pyFunc_;

    Point topLeft_;
    Point bottomRight_;
};

int imgCropper::init(const string& scriptPath) {
    // 初始化Python
    Py_Initialize();
    // 检查初始化是否成功
    if (!Py_IsInitialized()) {
        return -1;
    }
    // 添加路径
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");
    string command = "sys.path.append('" + scriptPath +"')";
    PyRun_SimpleString(command.c_str());

    import_array(); 
    // cout << "Inited." << endl;
    return 0;
}

int imgCropper::findModule(const string& scriptName) {
    pyModule_ = PyImport_ImportModule(scriptName.c_str());
    if(!pyModule_) {
        printf("Can't find [%s].py!", scriptName.c_str());
        return -1;
    }
    // cout << "pyModule Found." << endl;
    return 0;
}

int imgCropper::findFunc(const string& funcName) {
    pyFunc_= PyObject_GetAttrString(pyModule_, funcName.c_str());
    if(!pyFunc_) {
        printf("Can't find function [%s]!", funcName.c_str());
        return -1;
    }
    if(!PyCallable_Check(pyFunc_)) {
        printf("Can't call function [%s]!", funcName.c_str());
        return -1;
    }
    // cout << "pyFunc Found." << endl;
    return 0;
}

cv::Rect imgCropper::getCroppedBox(const Mat& src) {
    assert(src.isContinuous());
    assert(pyModule_ && pyFunc_);

	// npy_intp dim_np[] = {src.rows, src.cols, src.channels()};
    npy_intp* dim_np = new npy_intp[]{src.rows, src.cols, src.channels()};

    // 构造参数
    // PyObject* PyArray = PyArray_SimpleNewFromData(sizeof(dim_np) / sizeof(npy_intp), dim_np, NPY_UBYTE, 
    //                                                 static_cast<void*>(src.data));
    PyObject* PyArray = PyArray_SimpleNewFromData(3, dim_np, NPY_UBYTE, 
                                                    static_cast<void*>(src.data));
    PyObject* ArgArray = PyTuple_New(1); //同样定义大小与Python函数参数个数一致的PyTuple对象
    PyTuple_SetItem(ArgArray, 0, PyArray); 
    PyObject* pyReturn = PyEval_CallObject(pyFunc_, ArgArray);

    PyArg_ParseTuple(pyReturn, "iiii", &topLeft_.x, &topLeft_.y, &bottomRight_.x, &bottomRight_.y);

    Py_DECREF(PyArray);
    Py_DECREF(ArgArray);
    Py_DECREF(pyReturn);
    delete[] dim_np;


    const int kGap = 20;
    topLeft_.x = topLeft_.x > kGap ? topLeft_.x - kGap : 0;
    topLeft_.y = topLeft_.y > kGap ? topLeft_.y - kGap : 0;

    bottomRight_.x = bottomRight_.x + kGap < src.cols ? bottomRight_.x + kGap : src.cols;
    bottomRight_.y = bottomRight_.y + kGap < src.rows ? bottomRight_.y + kGap : src.rows;


    int width = bottomRight_.x - topLeft_.x, height = bottomRight_.y - topLeft_.y;
    assert(width >= 0 && height >= 0);
    return {topLeft_.x, topLeft_.y, width, height};
}

imgCropper::~imgCropper() {
    Py_DECREF(pyModule_);
    Py_DECREF(pyFunc_);
    Py_Finalize();
}

#endif // IMGCROPPER_H