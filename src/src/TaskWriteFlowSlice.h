#ifndef TASK_WRITE_FLOW_SLICE_H
#define TASK_WRITE_FLOW_SLICE_H

#include <fstream>

#include <boost/format.hpp>

#include "zlib.h"

#include "IFlowSinkTask.h"
#include "FlowSlice.h"

#define TAG_STRING "PIEH"    // use this when WRITING the file with middlebury

class OutputPolicyTSV
{
protected:
	void write(std::shared_ptr<FlowSlice> flowSlice, const std::string fn_path, int sliceCount)
	{
		boost::format fmt("/%08d");
		std::ofstream filexv(fn_path+(fmt%sliceCount).str()+"_xv.tsv", std::ios::out);
		std::ofstream fileyv(fn_path+(fmt%sliceCount).str()+"_yv.tsv", std::ios::out);

		if(filexv.is_open() && fileyv.is_open())
		{
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
		if(filexv.is_open() && fileyv.is_open())
		{
			filexv.write((char*) flowSlice->xv_.data(), flowSlice->xv_.rows() * flowSlice->xv_.cols() * sizeof(float));
			fileyv.write((char*) flowSlice->yv_.data(), flowSlice->yv_.rows() * flowSlice->yv_.cols() * sizeof(float));
			filexv.close();
			fileyv.close();
		}
	}
};

class OutputPolicyCompressed
{
protected:
	void write(std::shared_ptr<FlowSlice> flowSlice, const std::string fn_path, int sliceCount)
	{
		boost::format fmt("/%08d");
		gzFile filexv = gzopen(std::string(fn_path+(fmt%sliceCount).str()+"_xv.ebflo.gz").c_str(), "wb");
		gzFile fileyv = gzopen(std::string(fn_path+(fmt%sliceCount).str()+"_yv.ebflo.gz").c_str(), "wb");
		gzwrite(filexv, flowSlice->xv_.data(), flowSlice->xv_.rows() * flowSlice->xv_.cols() * sizeof(float));
		gzwrite(fileyv, flowSlice->yv_.data(), flowSlice->yv_.rows() * flowSlice->yv_.cols() * sizeof(float));
		gzclose(filexv);
		gzclose(fileyv);
	}
};

class OutputMiddlebury
{
protected:
	void write(std::shared_ptr<FlowSlice> flowSlice, const std::string fn_path, int sliceCount)
	{
		boost::format fmt("/%08d");
		std::string fn = std::string(fn_path+(fmt%sliceCount).str()+".flo");

		int cols = flowSlice->xv_.cols();
		int rows = flowSlice->xv_.rows();
		//Write Header
		std::ofstream file(fn.c_str(), std::ios::out | std::ios::binary);
		if(file.is_open() && file.is_open())
		{
			file << "PIEH";
			file.write((char*) &cols, sizeof(int));
			file.write((char*) &rows, sizeof(int));

			for(int y = 0; y < rows; ++y)
			{
				for(int x = 0; x < cols; ++x)
				{
					file.write((char*) flowSlice->xv_.data() + y*cols + x, sizeof(float));
					file.write((char*) flowSlice->yv_.data() + y*cols + x, sizeof(float));
				}
			}

			file.close();
		}

		// FILE *stream = fopen(fn.c_str(), "wb");
		// if (stream == 0) return;

		// // write the header
		// fprintf(stream, TAG_STRING);
		// if ((int)fwrite(&cols,  sizeof(int),   1, stream) != 1 ||
		// 	(int)fwrite(&rows, sizeof(int),   1, stream) != 1)
		// 	return;

			// // write the rows
			// int n = nBands * width;
			// for (int y = 0; y < height; y++) {
			// 	float* ptr = &img.Pixel(0, y, 0);
			// 	if ((int)fwrite(ptr, sizeof(float), n, stream) != n)
			// 		return 0;
			// }

		// fclose(stream);
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