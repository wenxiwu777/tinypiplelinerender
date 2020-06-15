cmake_minimum_required(VERSION 3.15)
project(tgatest)

set(CMAKE_CXX_STANDARD 11)

file(GLOB SOURCES
        *.h
        *.cpp)

add_executable(tgatest main.cpp ${SOURCE_FILES} ${SOURCES})
