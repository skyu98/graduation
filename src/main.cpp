#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include "jar.h"
#include <Python.h>
#include <chrono>

using namespace std;
using namespace cv;


string input_dir = "../imgs/input_imgs/";
string output_dir = "../imgs/output_imgs/";

// int pythonFunc() {
//     // 初始化Python
//     //在使用Python系统前，必须使用Py_Initialize对其
//     //进行初始化。它会载入Python的内建模块并添加系统路
//     //径到模块搜索路径中。这个函数没有返回值，检查系统
//     //是否初始化成功需要使用Py_IsInitialized。
//     Py_Initialize();
 
//     // 检查初始化是否成功
//     if ( !Py_IsInitialized() ) {
//         return -1;
//     }
 
//     // 添加当前路径。这里注意下面三句都不可少，
//     //添加的是当前路径。但是我打印了sys.path,
//     //出来了好多路径，有点类似环境变量路径的东西。这点不太懂怎么就成当前路径了
//     PyRun_SimpleString("import sys");
//     PyRun_SimpleString("print '---import sys---'");
//     //下面这个./表示当前工程的路径，如果使用../则为上级路径，根据此来设置
//     PyRun_SimpleString("print sys.path.append('./')"); 
 
//     PyObject *pName,*pModule,*pDict,*pFunc,*pArgs;
 
//     // 载入名为your_file的脚本
//     pName = PyBytes_FromString("your_file");
//     pModule = PyImport_Import(pName);
//     if ( !pModule ) {
//         printf("can't find your_file.py");
//         getchar();
//         return -1;
//     }
 
//     pDict = PyModule_GetDict(pModule);
//     if ( !pDict ) {
//         return -1;
//     }
//     printf("----------------------\n");
 
//     // 找出函数名为display的函数
//     pFunc = PyDict_GetItemString(pDict, "display");
//     if ( !pFunc || !PyCallable_Check(pFunc) ) {
//         printf("can't find function [display]");
//         getchar();
//         return -1;
//      }
 
//     //将参数传进去。1代表一个参数。
//     pArgs = PyTuple_New(1);
 
//     //  PyObject* Py_BuildValue(char *format, ...)
//     //  把C++的变量转换成一个Python对象。当需要从
//     //  C++传递变量到Python时，就会使用这个函数。此函数
//     //  有点类似C的printf，但格式不同。常用的格式有
//     //  s 表示字符串，
//     //  i 表示整型变量，
//     //  f 表示浮点数，
//     //  O 表示一个Python对象。
//     //这里我要传的是字符串所以用s，注意字符串需要双引号！
//     PyTuple_SetItem(pArgs, 0, Py_BuildValue("s"," python in C++"));
//     // 调用Python函数
//     PyObject_CallObject(pFunc, pArgs);
 
//     // 关闭Python
//     Py_Finalize();
//     return 0;
// }

int main(int argc, char* argv[]) {
    string imgName = argc >=2 ? argv[1] : "1.jpg";

    auto start = std::chrono::high_resolution_clock::now();
    Jar jar;
    if(!jar.init(input_dir + imgName))
        return -1;

    jar.getPosture();
    jar.getObstruction();

    auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> timeUsed = end - start;
	cout << "Time used : " << timeUsed.count() << " ms" << endl;

    jar.drawResult(output_dir + imgName);
    return 0;
}