# name of executable file and path of source file
add_executable(main utils.cpp main.cpp jar.cpp)
# opencv libraries
target_link_libraries(main ${OpenCV_LIBS} python3.7m)

# add_subdirectory(test)
# add_subdirectory(experiment)