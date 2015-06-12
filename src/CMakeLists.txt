cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project( optical_flow CXX)

option(WITH_CUDA "Enable CUDA backend" ON)

#    Set CMake Install Path
set (CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "default install path" FORCE )

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
add_definitions ("-Wall -Wno-deprecated-declarations")

#   Debug/Profiling Options
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

#	CMake module path for custom module finding
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/modules)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

#-------------------
# Edvstools Build
#-------------------
set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)
add_subdirectory(${EXT_PROJECTS_DIR}/edvstools)
include_directories(${EDVSTOOLS_INCLUDE_DIR})
link_directories(${EDVSTOOLS_LIB_DIR})

# Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
	configure_file(
		${CMAKE_SOURCE_DIR}/../docs/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
		add_custom_target(doc
		${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generating API documentation with Doxygen" VERBATIM
	)
endif(DOXYGEN_FOUND)


# CUDA
find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "Building CUDA backend")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
elseif()
    message(STATUS "Could not find CUDA. Disabling CUDA backend")
    set(WITH_CUDA OFF)
endif()

add_subdirectory(exe)
add_subdirectory(src)
add_subdirectory(test)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
