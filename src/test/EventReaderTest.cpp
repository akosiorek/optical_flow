#include "gtest/gtest.h"
#include "common.h"
#include "Edvs/event.h"
#include "../src/BlockingDeque.h"
#include "EventReader.h"


class EventReaderTest : public testing::Test {

    void SetUp() {
        event_reader = std::make_unique<EventReader<BlockingDeque<Edvs::Event>>>();
    }

public:
    std::unique_ptr<EventReader<BlockingDeque<Edvs::Event>>> event_reader;
};

TEST_F(EventReaderTest, ConstructorTest) {

    ASSERT_EQ(event_reader->isBufferSet(), false);
}

TEST_F(EventReaderTest, SetGetURITest)
{
    event_reader->setURI("/dev/ttyUSB0?baudrate=4000000&dtsm=2&htsm=1&msmode=0");
    ASSERT_STREQ("/dev/ttyUSB0?baudrate=4000000&dtsm=2&htsm=1&msmode=0",
                 event_reader->getURI().c_str());
}

TEST_F(EventReaderTest, SetBufferTest)
{
    ASSERT_EQ(event_reader->isBufferSet(), false);
    event_reader->setOutputBuffer(std::make_shared<BlockingDeque<Edvs::Event>>());
    ASSERT_EQ(event_reader->isBufferSet(), true);
}

TEST_F(EventReaderTest, BadURITest)
{
    ASSERT_EQ(event_reader->isPublishing(), false);
    event_reader->startPublishing();
    ASSERT_EQ(event_reader->isPublishing(), false);
}
