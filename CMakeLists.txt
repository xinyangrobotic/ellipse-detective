cmake_minimum_required(VERSION 3.1)
project(fast_ellipse_detector)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        )
add_executable(fast_ellipse_detector
        common.cpp
        common.h
        EllipseDetectorYaed.cpp
        EllipseDetectorYaed.h
        Main.cpp
        )

target_link_libraries(fast_ellipse_detector
        ${OpenCV_LIBRARIES}
        )
