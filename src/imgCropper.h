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

    int init();
    int findModule(const string& scriptName);
    int findFunc(const string& funcName);

    pair<Point, Point> getCroppedBox(const Mat& src);

    ~imgCropper();
private:
    PyObject* pyModule_;
    PyObject* pyFunc_;

    Point topLeft_;
    Point bottomRight_;
};

int imgCropper::init() {
    // 初始化Python
    Py_Initialize();
    // 检查初始化是否成功
    if (!Py_IsInitialized()) {
        return -1;
    }
    // 添加路径
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')"); 
    PyRun_SimpleString("sys.path.append('../src/test')");

    import_array(); 
    cout << "Inited." << endl;
    return 0;
}

int imgCropper::findModule(const string& scriptName) {
    pyModule_ = PyImport_ImportModule(scriptName.c_str());
    if(!pyModule_) {
        printf("Can't find [%s].py!", scriptName.c_str());
        return -1;
    }
    cout << "pyModule Found." << endl;
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
    cout << "pyFunc Found." << endl;
    return 0;
}

pair<Point, Point> imgCropper::getCroppedBox(const Mat& src) {
    assert(src.isContinuous());
    assert(pyModule_ && pyFunc_);

	npy_intp dim_np[]{src.rows, src.cols, src.channels()};
    // 构造参数
    PyObject* PyArray  = PyArray_SimpleNewFromData(sizeof(dim_np) / sizeof(npy_intp), dim_np, NPY_UBYTE, 
                                                    static_cast<void*>(src.data));
    PyObject* ArgArray = PyTuple_New(1); //同样定义大小与Python函数参数个数一致的PyTuple对象
    PyTuple_SetItem(ArgArray, 0, PyArray); 
    PyObject* pyReturn = PyEval_CallObject(pyFunc_, ArgArray);

    PyArg_ParseTuple(pyReturn, "iiii", &topLeft_.x, &topLeft_.y, &bottomRight_.x, &bottomRight_.y);
    
    Py_DECREF(PyArray);
    Py_DECREF(ArgArray);
    Py_DECREF(pyReturn);
    return {topLeft_, bottomRight_};
}

imgCropper::~imgCropper() {
    Py_DECREF(pyModule_);
    Py_DECREF(pyFunc_);
    Py_Finalize();
}

#endif // IMGCROPPER_H