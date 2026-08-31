cmake_minimum_required(VERSION 3.22.0)
set(lib_name NNdeployment)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(NNDEPLOYMENT_ENABLE_TENSORRT "Build TensorRT backend support" OFF)

find_package(OpenCV REQUIRED)


set(NNDEPLOYMENT_WITH_OPENVINO ON)
set(NNDEPLOYMENT_WITH_TENSORRT OFF)
if(NNDEPLOYMENT_ENABLE_TENSORRT)
    set(NNDEPLOYMENT_WITH_OPENVINO OFF)
    set(NNDEPLOYMENT_WITH_TENSORRT ON)
    find_package(CUDAToolkit REQUIRED)
    # TensorRT 没有稳定的官方 CMake package，这里手动查找头文件和库
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS /usr/include/x86_64-linux-gnu
              /usr/src/tensorrt/include
              /usr/local/TensorRT-10.0.1.6/include/
        REQUIRED
    )
    find_library(TENSORRT_LIBRARY nvinfer
        PATHS /usr/lib/x86_64-linux-gnu
              /usr/src/tensorrt/lib
              /usr/local/TensorRT-10.0.1.6/lib/
        REQUIRED
    )
else()
    find_package(OpenVINO REQUIRED)
endif()



set(NNdeployment_src
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/network_deployment_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/network_deployment_lib.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/network_preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess/network_postprocess.cpp
)

if(NNDEPLOYMENT_WITH_TENSORRT)
    list(APPEND NNdeployment_src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/network_deployment_tensorrt.cpp
    )
else()
    list(APPEND NNdeployment_src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/inference/network_deployment_openvino.cpp
    )
endif()

add_library(${lib_name}_lib SHARED ${NNdeployment_src})

add_library(${lib_name}_mpt_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/performance/network_performance.cpp
)
set_target_properties(${lib_name}_mpt_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${lib_name}_mpt_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/performance
)

# 头文件路径
target_include_directories(${lib_name}_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/interface
    ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(${lib_name}_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/preprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/postprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/performance
)

# 链接库
target_link_libraries(${lib_name}_lib PUBLIC
    ${OpenCV_LIBS}
)
target_link_libraries(${lib_name}_lib PRIVATE ${lib_name}_mpt_lib)

if(NNDEPLOYMENT_WITH_OPENVINO)
    target_link_libraries(${lib_name}_lib PRIVATE
        openvino::runtime
    )
endif()

if(NNDEPLOYMENT_WITH_TENSORRT)
    target_include_directories(${lib_name}_lib PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
    )
    target_link_libraries(${lib_name}_lib PRIVATE
        ${TENSORRT_LIBRARY}
        CUDA::cudart
    )
    target_compile_definitions(${lib_name}_lib PRIVATE NNDEPLOYMENT_WITH_TENSORRT=1)
else()
    target_compile_definitions(${lib_name}_lib PRIVATE NNDEPLOYMENT_WITH_TENSORRT=0)
endif()

if(NNDEPLOYMENT_WITH_OPENVINO)
    target_compile_definitions(${lib_name}_lib PRIVATE NNDEPLOYMENT_WITH_OPENVINO=1)
else()
    target_compile_definitions(${lib_name}_lib PRIVATE NNDEPLOYMENT_WITH_OPENVINO=0)
endif()
