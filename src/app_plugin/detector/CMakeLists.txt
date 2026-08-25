cmake_minimum_required(VERSION 3.22)

set(lib_name NNdetect)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui)

add_library(${lib_name}_lib SHARED
    src/NNDetector.cpp
    src/DetectorRuntime.cpp
)
target_compile_features(${lib_name}_lib PUBLIC cxx_std_20)
target_compile_definitions(${lib_name}_lib PRIVATE
    NETLIB_PROJECT_ROOT="${PROJECT_SOURCE_DIR}"
)
target_include_directories(${lib_name}_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(${lib_name}_lib
    PUBLIC
        NNdeployment_lib
    PRIVATE
        opencv_core
        opencv_imgproc
        opencv_videoio
)

add_executable(main examples/main.cpp)
target_link_libraries(main PRIVATE
    ${lib_name}_lib
    opencv_core
    opencv_highgui
    opencv_videoio
)

add_executable(main_test_rune tests/main_test_rune.cpp)
target_link_libraries(main_test_rune PRIVATE
    ${lib_name}_lib
    opencv_core
    opencv_highgui
    opencv_videoio
)

add_executable(main_test_speed tests/main_test_speed.cpp)
target_link_libraries(main_test_speed PRIVATE
    ${lib_name}_lib
    NNdeployment_mpt_lib
    opencv_core
    opencv_videoio
)
