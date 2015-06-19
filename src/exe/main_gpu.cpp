#include "common.h"
#include "utils.h"
#include "types.h"
#include "BlockingQueue.h"
#include "EventReader.h"
#include "Quantizer.h"
#include "FilterFactory.h"
#include "FourierPadder.h"
#include "FourierTransformerCUFFTW.h"
#include "FilteringEngineGPU.h"

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
    auto transformer = std::make_unique<FourierTransformerCUFFTW>(padder->fourierSizeRows_,
                                                                padder->fourierSizeCols_);

    FilteringEngineGPU<QueueT, QueueT> engine(std::move(factory), std::move(padder), std::move(transformer));

    engine.setInputBuffer(eventSliceQueue);
    engine.setOutputBuffer(flowSliceQueue);
    for(auto angle : cfg.filterAngles) {
        engine.addFilter(angle);
    }

    //TODO implement FlowSink
    FlowSinkProcessor<QueueT> sink;
    sink.setInputBuffer(flowSliceQueue);
    auto ebfloWriter = std::make_unique<TaskWriteFlowSlice<OutputPolicyBinary> >();
    ebfloWriter->setFilePath(cfg.fn_path);
    sink.addTask(std::move(ebfloWriter));

    // Start Processing
    LOG(INFO) << "Initialization completed";
    LOG(INFO) << "Processing...";

    boost::timer::auto_cpu_timer t;
    if(eventReader.startPublishing())
    {
        // TODO handle keyboard interrupts
        while(eventReader.isPublishing() || !eventQueue->empty())
        {
            quantizer.process();
            engine.process();
            if(flowSliceQueue->size() > 10) break;
        }
    }
    LOG(INFO) << "Processing finished. Completed " << flowSliceQueue->size() << " FlowSlices!";

    sink.start();
    while(!flowSliceQueue->empty())
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    sink.stop();

    //shutdown?
    eventReader.stopPublishing();

	return 0;
}
