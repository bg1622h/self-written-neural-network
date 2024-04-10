cmake_minimum_required(VERSION 3.24)
project(neur)

set(CMAKE_CXX_STANDARD 17)
SET(MYCOMPILE_FLAGS "-O3") # без O3 это будет вечно
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${MYCOMPILE_FLAGS}")

add_executable(neur main.cpp
        Tensor.h
        init.h
        layer.h
        ConvLayer.h
        ReluLayer.h
        maxpooling.h
        fullconectLayer.h
        tripletlosslayer.h
        ReluLayer.cpp
        Tensor.cpp
        ConvLayer.cpp
        maxpooling.cpp
        fullconectLayer.cpp
        tripletlosslayer.cpp
        )

find_package(OpenCV REQUIRED)
target_link_libraries(neur ${OpenCV_LIBS})