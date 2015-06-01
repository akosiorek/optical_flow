cmake_minimum_required(VERSION 2.8)
project( src )

add_library(optical
		Filter.cpp
		FilterFactory.cpp
		IFourierTransformer.cpp
		FourierTransformerFFTW.cpp
	)

add_executable(exe main.cpp)
target_link_libraries(exe optical ${GLOG_LIBRARIES} ${Boost_LIBRARIES} fftw3f ${EDVSTOOLS_LIB})