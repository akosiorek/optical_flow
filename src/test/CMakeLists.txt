
add_subdirectory(gtest)

add_executable(test.bin
    QuantizerTest.cpp
    BlockingQueueTest.cpp
    FourierPadderTest.cpp
    FilterTest.cpp
    FilterFactoryTest.cpp
    FilteringEngineCPUTest.cpp
    FilteringEngineGPUTest.cpp
    FourierTransformerTest.cpp
    EventReaderTest.cpp
    )

target_link_libraries(test.bin PRIVATE gtest_main optical_flow)
add_custom_command(TARGET test.bin POST_BUILD COMMAND test.bin)
