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

	EventReader() 
	: 	bufferSet_(false),
		buffer_(nullptr),
		uri_(std::string("")),
		stream_(nullptr),
		eventPublisher_(nullptr),
		running_(false) {}
	~EventReader()
	{
	    stopPublishing();
	}


	void setURI(const std::string& uri) { LOG_FUN; uri_ = uri; }
	std::string getURI() const { LOG_FUN; return uri_; }
	bool isURISet() { return !uri_.empty(); }

	void setOutputBuffer(std::shared_ptr<BufferType> buffer)
	{
		LOG_FUN;
		buffer_ = buffer;
		bufferSet_ = true;
	}

	bool isBufferSet() { LOG_FUN; return bufferSet_; }

	/**
	 * @brief Opens stream and start event polling thread
	 * @details [long description]
	 */
	bool startPublishing()
	{
		LOG_FUN_START;
		if(!isBufferSet())
		{
			std::cerr << "No event buffer has been set!" << std::endl;
			return false;
		}

		if(!isURISet())
		{
			std::cerr << "URI has not been set!" << std::endl;
			return false;
		}

		if(!openStream())
		{
			std::cerr << "Stream could not be opened!" << std::endl;
			return false;
		}


		eventPublisher_ = std::make_unique<std::thread>(&EventReader::pollEventStream, this);

		if(eventPublisher_ != nullptr) return running_ = true, running_;
		else return false;
		LOG_FUN_END;
	}

	/**
	 * @brief Stop polling the event stream, shutsdown all threads
	 */
	void stopPublishing()
	{
		LOG_FUN;
		if(running_)
		{
			running_ = false;
			eventPublisher_->join();
			stream_.reset(); //guess this should trigger the shutdown of the edvs library
		}
	}

	bool isPublishing() { LOG_FUN; return (running_ && !stream_->eos()); }

private:
	/**
	 * @brief Opens stream and returns a bool if successful.
	 */
	bool openStream()
	{
		LOG_FUN;
		stream_ = Edvs::OpenEventStream(uri_);
		return stream_->is_open();
	}

	/**
	 * @brief Polls for new events and pushes them to buffer
	 */
	void pollEventStream()
	{
		LOG_FUN_START;
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
		running_ == false; // only of interest if stream->eos() notifies EOF
		LOG_FUN_END;
	}


	bool bufferSet_;
	std::shared_ptr<BufferType> buffer_;

	std::string uri_;

	std::shared_ptr<Edvs::IEventStream> stream_;
	std::unique_ptr<std::thread> eventPublisher_;
	std::atomic_bool running_;
};

#endif //EVENT_READER_H
