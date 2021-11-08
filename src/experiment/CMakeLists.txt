add_executable(highLight_removal highLight_removal.cpp ${PROJECT_SOURCE_DIR}/src/utils.cpp)
target_link_libraries(highLight_removal ${OpenCV_LIBS})
