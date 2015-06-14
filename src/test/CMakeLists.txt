
add_subdirectory(gtest)

set(TEST_SRC
    CommonMocks.cpp
    QuantizerTest.cpp
    BlockingQueueTest.cpp
    FourierPadderTest.cpp
    FilterTest.cpp
    FilterFactoryTest.cpp
    FilteringEngineCPUTest.cpp
    FourierTransformerCPUTest.cpp
    EventReaderTest.cpp
)

if(WITH_CUDA)
    set(TEST_SRC ${TEST_SRC}
        DeviceBlobTest.cpp
        FilteringEngineGPUTest.cpp
        FourierTransformerGPUTest.cpp
        GpuMathTest.cpp
    )
endif()


add_executable(test.bin ${TEST_SRC})
target_link_libraries(test.bin PRIVATE gtest_main optical_flow)
add_custom_command(TARGET test.bin POST_BUILD COMMAND test.bin)


