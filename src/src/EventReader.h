#ifndef EVENT_READER_H
#define EVENT_READER_H

#include <string>
#include <thread>
#include <atomic>

#include "Edvs/EventStream.hpp"

#include "utils.h"

/**
 * @brief Provide easy access to a stream of event
 * @details [long description]
 * @return [description]
 */
template<typename BufferType>
class EventReader
{
public:

	EventReader() : bufferSet_(false), uri_(std::string("")), running_(false) {}
	~EventReader() {}

	void setURI(const std::string& uri) { uri_ = uri; }
	std::string getURI() const { return uri_; }
	
	void setOutputBuffer(std::shared_ptr<BufferType> buffer)
	{
		buffer_ = buffer;
		bufferSet_ = true;
	}

	bool isBufferSet() { return bufferSet_; }

	/**
	 * @brief Opens stream and start event polling thread
	 * @details [long description]
	 */
	bool startPublishing()
	{
		if(!openStream())
		{
			std::cout << "stream could not be opened!" << std::endl;
			return false;
		}

		eventPublisher_ = std::make_unique<std::thread>(&EventReader::pollEventStream, this);

		if(eventPublisher_ != nullptr) return running_ = true, running_;
		else return false;
	}

	void stopPublishing()
	{
		if(running_)
		{
			running_ = false;
			eventPublisher_->join();
			stream_.reset(); //guess this should trigger the shutdown of the edvs library
		}
	}

	bool isPublishing() { return running_; }

private:
	/**
	 * @brief Opens stream and returns a bool if successful.
	 */
	bool openStream()
	{
		stream_ = Edvs::OpenEventStream(uri_);
		return stream_->is_open();
	}

	/**
	 * @brief Polls for new events and pushes them to buffer
	 */
	void pollEventStream()
	{
	    // capture events (run until end of file or Ctrl+C)
		while(!stream_->eos() && running_ == true)
		{
			// read events from stream
			auto events = stream_->read();
			if(!events.empty())
			{
				for(auto event : events)
				{
					buffer_->push(event);
				}
			}
		}
	}


	bool bufferSet_;
	std::shared_ptr<BufferType> buffer_;

	std::string uri_;

	std::shared_ptr<Edvs::IEventStream> stream_;
	std::unique_ptr<std::thread> eventPublisher_;
	std::atomic_bool running_;
};

#endif //EVENT_READER_H
