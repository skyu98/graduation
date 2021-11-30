add_executable(canny canny.cpp)
target_link_libraries(canny ${OpenCV_LIBS})

add_executable(test test.cpp ${PROJECT_SOURCE_DIR}/src/utils.cpp)
target_link_libraries(test ${OpenCV_LIBS})

add_executable(grabCutTest grabCutTest.cpp)
target_link_libraries(grabCutTest ${OpenCV_LIBS})

add_executable(getRGB getRGB.cpp)
target_link_libraries(getRGB ${OpenCV_LIBS})

add_executable(scan scan.cpp)
target_link_libraries(scan ${OpenCV_LIBS})

add_executable(callPython callPython.cpp)
target_link_libraries(callPython ${OpenCV_LIBS} python3.7m) # 

add_executable(findThreshold findThreshold.cpp)
target_link_libraries(findThreshold ${OpenCV_LIBS})