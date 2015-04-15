cmake_minimum_required(VERSION 2.8)
project( name_src )

add_executable(exe main.cpp)
target_link_libraries(exe ${GLOG_LIBRARIES})
