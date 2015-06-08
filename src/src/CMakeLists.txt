
find_package(Glog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system timer program_options REQUIRED)
find_package(FFTW3 COMPONENTS single REQUIRED)

add_library(optical_flow
    Filter.cpp
    FilterFactory.cpp
    IFourierTransformer.cpp
    FourierTransformerFFTW.cpp
)
add_dependencies(optical_flow edvstools)
target_link_libraries(optical_flow
    PRIVATE ${GLOG_LIBRARIES} ${Boost_LIBRARIES} ${EDVSTOOLS_LIB} ${FFTW3_LIBRARIES}
)

target_include_directories(optical_flow
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR}
    PRIVATE ${GLOG_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${FFTW3_INCLUDE_DIRS}
)