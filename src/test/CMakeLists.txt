add_executable(canny canny.cpp)
target_link_libraries(canny ${OpenCV_LIBS})

add_executable(test test.cpp)
target_link_libraries(test ${OpenCV_LIBS})

add_executable(grabCutTest grabCutTest.cpp)
target_link_libraries(grabCutTest ${OpenCV_LIBS})