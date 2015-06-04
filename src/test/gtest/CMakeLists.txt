project(gtest CXX C)
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

find_package(Threads)
find_package(Glog)

add_library(gtest gtest-all.cpp)
target_link_libraries(gtest PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(gtest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)

add_library(gtest_main gtest_main.cc)
target_link_libraries(gtest_main PUBLIC gtest ${GLOG_LIBRARIES})