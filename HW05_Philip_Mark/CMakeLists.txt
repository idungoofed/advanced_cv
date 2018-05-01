cmake_minimum_required(VERSION 3.5)

# part A and B
set(PROJECT_NAME HW05_Stereo_Analysis_partA_B)
set(SOURCE_FILES stereo_analysis_partA_B.cpp)
project(${PROJECT_NAME})
find_package(OpenCV REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
