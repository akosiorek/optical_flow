#include <boost/program_options.hpp>

#include "common.h"
#include "types.h"
#include "BlockingQueue.h"
#include "EventReader.h"
#include "Quantizer.h"
#include "FilterFactory.h"
#include "FourierPadder.h"
#include "FourierTransformerFFTW.h"
#include "FilteringEngine.h"

template<class T>
using QueueT = BlockingQueue<T>;

int main(int argc, char** argv)
{
    std::string fn_input;
    int logLevel;

// BOOST  Program Options
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("filename,f", po::value<std::string>(&fn_input), "filename for event file / URI")
        ("loglevel,l", po::value<int>(&logLevel)->default_value(0),
                        "loglevel: INFO, WARNING, ERROR, and FATAL are 0, 1, 2, and 3")
    ;
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    if(vm.count("help") || !vm.count("filename")) {
        std::cout << desc << std::endl;
        return 1;
    }

// Setting up Loggin Systems
    google::InitGoogleLogging(argv[0]);

// Start Setup
    LOG(INFO) << "Event Based Optical Flow";
    LOG(INFO) << "Initializing...";

    //  Configuration params
    int timeSliceDuration = 100;
    int filterSize = 21;
    int dataSize = 128;

    float t0 = 0;
    float tk = 0.7;
    float timeResolution = timeSliceDuration * 1e-6f;
    int spatialRange = (filterSize - 1) / 2;

    std::vector<int> filterAngles = {0, 45, 90, 135};

    // TODO more logging
    LOG(INFO) << "Time slice duration: " << timeSliceDuration;
    LOG(INFO) << "Filter size: " << filterSize;


    // Buffers
    auto eventQueue = std::make_shared<QueueT<Event>>();
    auto eventSliceQueue = std::make_shared<QueueT<EventSlice::Ptr>>();
    auto flowSliceQueue = std::make_shared<QueueT<FlowSlice::Ptr>>();

    // Startup
    EventReader<QueueT<Event>> eventReader;
    eventReader.setOutputBuffer(eventQueue);

    Quantizer<QueueT> quantizer(timeSliceDuration);
    quantizer.setInputBuffer(eventQueue);
    quantizer.setOutputBuffer(eventSliceQueue);

    auto factory = std::make_unique<FilterFactory>(t0, tk, timeResolution, spatialRange, spatialRange);
    auto padder = std::make_unique<FourierPadder>(dataSize, filterSize);
    auto transformer = std::make_unique<FourierTransformerFFTW>(padder->fourierSizePadded_,
                                                                padder->fourierSizePadded_);

    FilteringEngine<QueueT, QueueT> engine(std::move(factory), std::move(padder), std::move(transformer));

    engine.setInputBuffer(eventSliceQueue);
    engine.setOutputBuffer(flowSliceQueue);
    for(auto angle : filterAngles) {
        engine.addFilter(angle);
    }

    //TODO implement FlowSink
    // FlowSink<QueueT> sink;
    // sink.setInputBuffer(flowSliceQueue);

// Start Processing
    LOG(INFO) << "Initialization completed";
    LOG(INFO) << "Processing...";


    eventReader.startPublishing();
    // TODO handle keyboard interrupts
    while(true) {
        quantizer.process();
        engine.process();
        // sink.process();
    }

	return 0;
}
