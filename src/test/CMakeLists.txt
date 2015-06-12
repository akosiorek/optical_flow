
add_subdirectory(gtest)

add_executable(test.cpu
    QuantizerTest.cpp
    BlockingQueueTest.cpp
    FourierPadderTest.cpp
    FilterTest.cpp
    FilterFactoryTest.cpp
    FilteringEngineCPUTest.cpp
    FourierTransformerTest.cpp
    EventReaderTest.cpp
)

target_link_libraries(test.cpu PRIVATE gtest_main optical_flow)
add_custom_command(TARGET test.cpu POST_BUILD COMMAND test.cpu)


if(WITH_CUDA)
    add_executable(test.gpu
        FilteringEngineGPUTest.cpp
        GpuMathTest.cpp
    )
    target_link_libraries(test.gpu PRIVATE gtest_main optical_flow)
    add_custom_command(TARGET test.gpu POST_BUILD COMMAND test.gpu)
endif()