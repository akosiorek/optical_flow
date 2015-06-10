#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>

#include "common.h"
#include "types.h"
#include "BlockingQueue.h"
#include "EventReader.h"
#include "Quantizer.h"
#include "FilterFactory.h"
#include "FourierPadder.h"
#include "FourierTransformerFFTW.h"
#include "FilteringEngineGPU.h"

template<class T>
using QueueT = BlockingQueue<T>;

int main(int argc, char** argv)
{
    std::string fn_input;

// BOOST  Program Options
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("filename,f", po::value<std::string>(&fn_input), "filename for event file / URI")
        ("loglevel", po::value<int>(&FLAGS_minloglevel)->default_value(FLAGS_minloglevel),
                        "loglevel: INFO, WARNING, ERROR, and FATAL are 0, 1, 2, and 3")
        ("logdir", po::value<std::string>(&FLAGS_log_dir)->default_value(FLAGS_log_dir),
                        "Location where log files will be saved")
        ("logtostderr", po::value<bool>(&FLAGS_logtostderr)->default_value(FLAGS_logtostderr),
                        "Log to stderr")
    ;
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    if(vm.count("help") || !vm.count("filename")) {
        std::cout << desc << std::endl;
        return 1;
    }

// Setting up Logging Systems
    google::InitGoogleLogging(argv[0]);

// Start Setup
    LOG(INFO) << "Event Based Optical Flow";
    LOG(INFO) << "Using " << Eigen::nbThreads() << " threads for Eigen.";
    LOG(INFO) << "Initializing...";

    //  Configuration params
    int timeSliceDuration = 10000;
    int filterSize = 21;
    int dataSize = 128;

    float t0 = 0;
    float tk = 0.7;
    float timeResolution = timeSliceDuration * 1e-6f;
    int spatialRange = (filterSize - 1) / 2;

    LOG(INFO) << "Parameters Setup:";
    LOG(INFO) << "t0: " << t0;
    LOG(INFO) << "t1: " << t0;
    LOG(INFO) << "timeResolution: " << timeResolution;
    LOG(INFO) << "spatialRange: " << spatialRange;

    LOG(INFO) << "timeSpan_ " << (tk - t0) / timeResolution;
    LOG(INFO) << "t0_ conv " << t0 / timeResolution;

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
    eventReader.setURI(fn_input);

    Quantizer<QueueT> quantizer(timeSliceDuration);
    quantizer.setInputBuffer(eventQueue);
    quantizer.setOutputBuffer(eventSliceQueue);

    auto factory = std::make_unique<FilterFactory>(t0, tk, timeResolution, spatialRange, spatialRange);
    auto padder = std::make_unique<FourierPadder>(dataSize, filterSize);
    auto transformer = std::make_unique<FourierTransformerFFTW>(padder->fourierSizeRows_,
                                                                padder->fourierSizeCols_);

    FilteringEngineGPU<QueueT, QueueT> engine(std::move(factory), std::move(padder), std::move(transformer));

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

    std::size_t lastFlowStatus = 0;
    int counter = 0;
    boost::timer::auto_cpu_timer t;
    if(eventReader.startPublishing())
    {
        // TODO handle keyboard interrupts
        while(true)
        {
            quantizer.process();
            engine.process();
            // sink.process();

            // Get out of this loop once there is no change in FlowSlices anymore!
            if((lastFlowStatus == flowSliceQueue->size()) && lastFlowStatus > 0)
            {
                if(counter > 5000) break;
                else ++counter;
            }

            lastFlowStatus = flowSliceQueue->size();
        }
    }

    LOG(INFO) << "Processing finished. Completed " << flowSliceQueue->size() << " FlowSlices!";
    //shutdown?
    eventReader.stopPublishing();

	return 0;
}
