cmake_minimum_required(VERSION 3.5)

# image rotation
set(PROJECT_NAME HW01_Image_Rotation)
set(SOURCE_FILES HW01_Image_Rotation.cpp)
project(${PROJECT_NAME})
find_package(OpenCV REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# video display
set(PROJECT_NAME HW01_VideoDisplay)
set(SOURCE_FILES HW01_VideoDisplay.cpp)
project(${PROJECT_NAME})
find_package(OpenCV REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# steganography
set(PROJECT_NAME HW01_Steganography)
set(SOURCE_FILES HW01_Steganography.cpp)
project(${PROJECT_NAME})
find_package(OpenCV REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# steganography
set(PROJECT_NAME testing)
set(SOURCE_FILES testing.cpp)
project(${PROJECT_NAME})
find_package(OpenCV REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})