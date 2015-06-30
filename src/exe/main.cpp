#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

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
#include "FilteringEngineCPU.h"

#include "FlowSinkProcessor.h"
#include "TaskWriteFlowSlice.h"

#include "runtimeHelpers.h"

template<class T>
using QueueT = BlockingQueue<T>;

int main(int argc, char** argv)
{
    EBOFConfig cfg;

    if(init(argc, argv, cfg)) return 1; // If help, fnc returns 1
    logParameters(cfg);

    // Buffers
    auto eventQueue = std::make_shared<QueueT<Event>>();
    auto eventSliceQueue = std::make_shared<QueueT<EventSlice::Ptr>>();
    auto flowSliceQueue = std::make_shared<QueueT<FlowSlice::Ptr>>();

    // Startup
    EventReader<QueueT<Event>> eventReader;
    eventReader.setOutputBuffer(eventQueue);
    eventReader.setURI(cfg.fn_input);

    Quantizer<QueueT> quantizer(cfg.timeSliceDuration);
    quantizer.setInputBuffer(eventQueue);
    quantizer.setOutputBuffer(eventSliceQueue);

    auto factory = std::make_unique<FilterFactory>(cfg.t0, cfg.tk, cfg.timeResolution, cfg.spatialRange, cfg.spatialRange);
    auto padder = std::make_unique<FourierPadder>(cfg.dataSize, cfg.filterSize);
    auto transformer = std::make_unique<FourierTransformerFFTW>(padder->fourierSizeRows_,
                                                                padder->fourierSizeCols_);

    FilteringEngineCPU<QueueT, QueueT> engine(std::move(factory), std::move(padder), std::move(transformer));

    engine.setInputBuffer(eventSliceQueue);
    engine.setOutputBuffer(flowSliceQueue);
    for(auto angle : cfg.filterAngles) {
        engine.addFilter(angle);
    }

    FlowSinkProcessor<QueueT> sink;
    sink.setInputBuffer(flowSliceQueue);
    auto ebfloWriter = std::make_unique<TaskWriteFlowSlice<OutputPolicyBinary> >();
    ebfloWriter->setFilePath(cfg.fn_path);
    sink.addTask(std::move(ebfloWriter));

    // Start Processing
    LOG(INFO) << "Initialization completed";
    LOG(INFO) << "Processing...";

    auto timer = std::make_shared<boost::timer::auto_cpu_timer>();
    if(eventReader.startPublishing())
    {
        // TODO handle keyboard interrupts
        while(eventReader.isPublishing() || !eventQueue->empty())
        {
            quantizer.process();
            engine.process();
        }
    }
    timer.reset();//trigger time printing
    LOG(INFO) << "Processing finished. Completed " << flowSliceQueue->size() << " FlowSlices!";

    if(!cfg.fn_path.empty())
    {
        FlowSinkProcessor<QueueT> sink;
        sink.setInputBuffer(flowSliceQueue);
        auto ebfloWriter = std::make_unique<TaskWriteFlowSlice<OutputPolicyBinary> >();
        ebfloWriter->setFilePath(cfg.fn_path);
        sink.addTask(std::move(ebfloWriter));
        sink.start();
        while(!flowSliceQueue->empty())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        sink.stop();
    }

    eventReader.stopPublishing();

	return 0;
}
