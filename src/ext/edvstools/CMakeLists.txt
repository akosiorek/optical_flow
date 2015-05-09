
cmake_minimum_required(VERSION 2.8.0)
project(edvstools)
include(ExternalProject)

ExternalProject_Add(edvstools
    GIT_REPOSITORY https://github.com/Danvil/edvstools.git
    CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
     PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
# Disable install step
    INSTALL_COMMAND ""
)

# Specify include dir
ExternalProject_Get_Property(edvstools source_dir)
set(EDVSTOOLS_INCLUDE_DIR ${source_dir} PARENT_SCOPE) # force use to type Edvs/***.h

# Specify edvs tools library dir and lib
ExternalProject_Get_Property(edvstools binary_dir)
# set(EDVSTOOLS_LIBS_DIR ${binary_dir} PARENT_SCOPE)
set(EDVSTOOLS_LIB ${binary_dir}/Edvs PARENT_SCOPE)