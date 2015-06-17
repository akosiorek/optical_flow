cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project( optical_flow CXX)

option(WITH_CUDA "Enable CUDA backend" OFF)

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

# Latex
FIND_PACKAGE(LATEX)
IF(LATEX_COMPILER)
set(REPORT_DIR ${CMAKE_SOURCE_DIR}/../report)
	add_custom_target(reportdvi
	COMMAND   ${LATEX_COMPILER} ${REPORT_DIR}/ebof.tex 
	DEPENDS   ${REPORT_DIR}/ebof.tex
	COMMENT   "Tex2dvi"
	)

	IF(DVIPS_CONVERTER)
		add_custom_target(reportps
		  COMMAND   ${DVIPS_CONVERTER} ${CMAKE_CURRENT_BINARY_DIR}/ebof.dvi
		            -o ${CMAKE_CURRENT_BINARY_DIR}/ebof.ps
		  DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/ebof.dvi
		  COMMENT   "dvi2ps"
		)
		add_dependencies(reportps reportdvi)

		IF(PS2PDF_CONVERTER)
			add_custom_target(report
			COMMAND   ${PS2PDF_CONVERTER} -sPAPERSIZE=a4 ${CMAKE_CURRENT_BINARY_DIR}/ebof.ps
			DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/ebof.ps
			COMMENT   "ps2pdf"
			)
			add_dependencies(report reportps)
		ENDIF(PS2PDF_CONVERTER)
	ENDIF(DVIPS_CONVERTER)
ENDIF(LATEX_COMPILER)


# CUDA
if(WITH_CUDA)
	find_package(CUDA)
	if(CUDA_FOUND)
	    message(STATUS "Building CUDA backend")
	    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	    set(CUDA_CUFFTW_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufftw.so)
	elseif()
	    message(STATUS "Could not find CUDA. Disabling CUDA backend")
	    set(WITH_CUDA OFF)
	endif()
endif()

add_subdirectory(exe)
add_subdirectory(src)
add_subdirectory(test)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
