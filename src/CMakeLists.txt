# name of executable file and path of source file
add_executable(main utils.cpp main.cpp jar.cpp Darknet.cpp imageCropper.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET main PROPERTY CXX_STANDARD 11)

add_executable(traverseAll utils.cpp traverseAll.cpp jar.cpp Darknet.cpp imageCropper.cpp)
target_link_libraries(traverseAll "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET traverseAll PROPERTY CXX_STANDARD 11)

add_subdirectory(test)
# add_subdirectory(experiment)