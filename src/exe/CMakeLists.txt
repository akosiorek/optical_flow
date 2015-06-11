

add_executable(optical_flow.cpu main.cpp)
target_link_libraries(optical_flow.cpu PRIVATE optical_flow)

#add_executable(optical_flow.gpu main_gpu.cpp)
#target_link_libraries(optical_flow.gpu PRIVATE optical_flow)