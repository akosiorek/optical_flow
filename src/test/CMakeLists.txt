project(Test)

# Compile gtest
add_subdirectory(gtest)
include_directories( ${gtest_INCLUDE_DIR})

include_directories(${src_SOURCE_DIR})

add_executable(test test.cpp 
					QuantizerTest.cpp 
					SequentialBufferTest.cpp 
					BlockingQueueTest.cpp
					FourierPadderTest.cpp)
target_link_libraries(test gtest_main optical)
add_custom_command(TARGET test POST_BUILD COMMAND test)
