#include <memory>

#include <glog/logging.h>
#include "gtest/gtest.h"

#include "utils.h"

#include "../src/EventReader.h"


class EventReaderTest : public testing::Test {

    void SetUp() {
        event_reader = make_unique<EventReader>();
    }

public:
    std::unique_ptr<EventReader> event_reader;
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
    event_reader->setBuffer(std::make_shared<Buffer>());
    ASSERT_EQ(event_reader->isBufferSet(), true);
}

TEST_F(EventReaderTest, BadURITest)
{
    ASSERT_EQ(event_reader->isPublishing(), false);
    event_reader->startPublishing();
    ASSERT_EQ(event_reader->isPublishing(), false);
}

// TEST_F(QuantizerTest, ConstructorTest) {

//     ASSERT_EQ(quantizer->getTimeResolution(), 2);
//     ASSERT_EQ(quantizer->getCurrentTimeStep(), 0);
// }

// TEST_F(QuantizerTest, NoEventTest) {

//     ASSERT_TRUE(quantizer->isEmpty());
//     quantizer->quantize(std::vector<Event>());
//     ASSERT_TRUE(quantizer->isEmpty());
//     ASSERT_TRUE(quantizer->getEventSlice().isZero(0));
// }

// TEST_F(QuantizerTest, SingleEventTest) {

//     // if time resolution > 1 there's not event slice with a single event.
//     // Another event, with a time + timeResolution, would have to be passed
//     // to generate an event slice.
//     std::vector<Event> events = {{1, 1, 1, 1}};
//     quantizer->quantize(events);
//     ASSERT_TRUE(quantizer->isEmpty());

//     auto eventSlice = quantizer->getEventSlice();
//     ASSERT_TRUE(eventSlice.isZero(0));

//     events = {{1, 2, 3, 4}};
//     quantizer->quantize(events);
//     ASSERT_FALSE(quantizer->isEmpty());
//     eventSlice = quantizer->getEventSlice();
//     ASSERT_EQ(eventSlice(1, 1), 1);
// }

// TEST_F(QuantizerTest, QuantizeTest) {
    
//     std::vector<Event> events = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {5, 5, 5, 5}};
//     quantizer->quantize(events);
//     ASSERT_FALSE(quantizer->isEmpty());

//     auto event = quantizer->getEventSlice();
//     ASSERT_FALSE(quantizer->isEmpty());
//     ASSERT_EQ(event(1, 1), 1);
//     ASSERT_EQ(event(2, 2), 2);
//     ASSERT_EQ(event(127, 127), 0);

//     event = quantizer->getEventSlice();
//     ASSERT_TRUE(quantizer->isEmpty());
//     ASSERT_EQ(event(1, 1), 0);
//     ASSERT_EQ(event(3, 3), 3);

//     ASSERT_TRUE(quantizer->getEventSlice().isZero(0));
// }

// TEST_F(QuantizerTest, QuantizeGetMultipleEventsTest) {

//     quantizer = make_unique<Quantizer>(3);
//     std::vector<Event> events = {
//             {1, 1, 1, 1},
//             {2, 2, 2, -1},
//             {3, 3, 3, 1},
//             {4, 4, 4, -1},
//             {5, 5, 5, 1},
//             {6, 6, 9, -1},
//             {7, 7, 10, 1},
//             {8, 8, 11, -1},
//             {9, 9, 12, 1},
//             {10, 10, 15, -1}
//     };
//     quantizer->quantize(events);

//     auto eventSlices = quantizer->getEventSlices();
//     ASSERT_TRUE(quantizer->isEmpty());
//     ASSERT_EQ(eventSlices.size(), 5);
// }

