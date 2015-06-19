#ifndef TASK_WRITE_FLOW_SLICE_H
#define TASK_WRITE_FLOW_SLICE_H

#include <fstream>

#include <boost/format.hpp>

#include "IFlowSinkTask.h"
#include "FlowSlice.h"

class OutputPolicyTSV
{
protected:
	void write(std::shared_ptr<FlowSlice> flowSlice, const std::string fn_path, int sliceCount)
	{
		boost::format fmt("/%08d");
		std::ofstream filexv(fn_path+(fmt%sliceCount).str()+"_xv.tsv", std::ios::out);
		std::ofstream fileyv(fn_path+(fmt%sliceCount).str()+"_yv.tsv", std::ios::out);

		auto ptrxv = flowSlice->xv_.data();
		auto ptryv = flowSlice->yv_.data();

		for(int i = 0; i<flowSlice->xv_.rows(); ++i) // ROWS
		{
			for(int j = 0; j<flowSlice->xv_.cols(); ++j) // COLS
			{
				filexv << *(ptrxv + j + i*flowSlice->xv_.cols());
				fileyv  << *(ptryv + j + i*flowSlice->xv_.cols());
				if(j<flowSlice->xv_.cols()-1)
				{
					filexv << "\t";
					fileyv << "\t";
				}
			}
			filexv << "\n";
			fileyv << "\n";
		}

		filexv.close();
		fileyv.close();
	}
};

class OutputPolicyBinary
{
protected:
	void write(std::shared_ptr<FlowSlice> flowSlice, const std::string fn_path, int sliceCount)
	{
		boost::format fmt("/%08d");
		std::ofstream filexv(fn_path+(fmt%sliceCount).str()+"_xv.ebflo", std::ios::out | std::ios::binary);
		std::ofstream fileyv(fn_path+(fmt%sliceCount).str()+"_yv.ebflo", std::ios::out | std::ios::binary);
		filexv.write((char*) flowSlice->xv_.data(), flowSlice->xv_.rows() * flowSlice->xv_.cols() * sizeof(float));
		fileyv.write((char*) flowSlice->yv_.data(), flowSlice->yv_.rows() * flowSlice->yv_.cols() * sizeof(float));
		filexv.close();
		fileyv.close();
	}
};


template<typename OutputPolicy>//, typename Type> //whereace type is angle/magnitude pairs oder xv,yv seperat
class TaskWriteFlowSlice: public IFlowSinkTask, private OutputPolicy
{
	using OutputPolicy::write;

public:
	TaskWriteFlowSlice()
		:	sliceCount_(0)
	{
		// DO SHIT
	}

	~TaskWriteFlowSlice() override {}

	void setFilePath(const std::string& path)
	{
		fn_path_ = path;
	}

	void process(std::shared_ptr<FlowSlice> flowSlice) override // TODO REPLACE WITH WRITE POLICY BINARY/TSV/BOOST SERIALIZATION
	{
		write(flowSlice, fn_path_, sliceCount_);
		++sliceCount_;
	}

private:
	int sliceCount_;
	std::string fn_path_;
};

#endif