# project name
PROJECT(graduation C CXX)
# requirement of cmake version
cmake_minimum_required(VERSION 2.8)

# if(NOT PYTHON_LIBRARY_DIRS)
#     find_package(Python 3 REQUIRED)
# endif()

set(CXX_FLAGS
 -g
 # -DVALGRIND
 -DCHECK_PTHREAD_RETURN_VALUE
 -D_FILE_OFFSET_BITS=64
 -Wall
 -Wextra
 # -Werror
 -Wconversion
 -Wno-unused-parameter
 -Wno-unused-variable
 -Wno-unused-result
 -Wno-shorten-64-to-32
 # -Wold-style-cast
 -Woverloaded-virtual
 -Wpointer-arith
 -Wshadow
 -Wwrite-strings
 -march=native
 # -MMD
 -std=c++11
 -pthread   # check
 -Wc++11-extensions
 )

string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CXX_FLAGS}")
# set the directory of executable files

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(Torch_DIR /Users/yuxiao/Projects/graduation/3rdParty/libtorch/share/cmake/Torch)
# set(Python_FIND_STRATEGY LOCATION)

# find required opencv
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(PythonInterp 3 REQUIRED)

# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Torch_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# include_directories(/Users/yuxiao/anaconda3/include/python3.7m)
# include_directories(/Users/yuxiao/anaconda3/lib/python3.7/site-packages/numpy/core/include)

# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# link_directories(${PYTHON_LIBRARY_DIRS})
# link_directories(/Users/yuxiao/anaconda3/lib)

add_subdirectory(src)
