# project name
PROJECT(graduation C CXX)
# requirement of cmake version
cmake_minimum_required(VERSION 2.8)

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

# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})

add_subdirectory(src)
