cmake_minimum_required(VERSION 3.22)

set(lib_name NNdetect)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

add_library(${lib_name}_lib SHARED
    src/NNDetector.cpp
)
target_compile_features(${lib_name}_lib PUBLIC cxx_std_20)
target_include_directories(${lib_name}_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(${lib_name}_lib
    PUBLIC
        NNdeployment_lib
    PRIVATE
        opencv_core
)

add_executable(main examples/main.cpp)
target_compile_definitions(main PRIVATE
    NETLIB_PROJECT_ROOT="${PROJECT_SOURCE_DIR}"
)
target_link_libraries(main PRIVATE
    ${lib_name}_lib
    opencv_core
    opencv_imgproc
    opencv_videoio
)

add_executable(main_test_speed tests/main_test_speed.cpp)
target_compile_definitions(main_test_speed PRIVATE
    NETLIB_PROJECT_ROOT="${PROJECT_SOURCE_DIR}"
)
target_link_libraries(main_test_speed PRIVATE
    ${lib_name}_lib
    NNdeployment_mpt_lib
    opencv_core
    opencv_videoio
)

if(BUILD_TESTING)
    add_executable(detector_smoke_test tests/detector_smoke_test.cpp)
    target_compile_definitions(detector_smoke_test PRIVATE
        NETLIB_PROJECT_ROOT="${PROJECT_SOURCE_DIR}"
    )
    target_link_libraries(detector_smoke_test PRIVATE
        ${lib_name}_lib
        opencv_core
        opencv_videoio
    )
    add_test(NAME detector_smoke COMMAND detector_smoke_test)
    set_tests_properties(detector_smoke PROPERTIES TIMEOUT 180)
endif()
