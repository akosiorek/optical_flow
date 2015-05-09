cmake_minimum_required(VERSION 2.8)
project( src )

add_executable(exe main.cpp)
target_link_libraries(exe ${GLOG_LIBRARIES} ${Boost_LIBRARIES} ${FFTW_LIBRARIES} )

add_library(optical
        Quantizer.cpp
        EventReader.cpp
        Filter.cpp
        FilterFactory.cpp
    	IFourierTransformer.cpp
    	FourierTransformerFFTW.cpp)
