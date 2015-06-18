#ifndef I_FLOW_SINK_TASK_H
#define I_FLOW_SINK_TASK_H

class FlowSlice;

class IFlowSinkTask
{
public:
	virtual ~IFlowSinkTask() {};
	virtual void process(std::shared_ptr<FlowSlice> flowSlice) {}
};

#endif //I_FLOW_SINK_TASK_H