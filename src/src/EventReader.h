#ifndef EVENT_READER_H
#define EVENT_READER_H

#include <string>
#include <thread>

#include "Edvs/event.h"
#include "Edvs/EventStream.hpp"

class Buffer
{
public:
	Buffer() {}
	~Buffer() {}
};

/**
 * @brief Provide easy access to a stream of event
 * @details [long description]
 * @return [description]
 */
class EventReader
{
public:
	EventReader() : bufferSet_(false), uri_(std::string("")), running_(false) {}
	~EventReader() {}

	void setURI(const std::string& uri) { uri_ = uri; }
	std::string getURI() const { return uri_; }

	void setBuffer(std::shared_ptr<Buffer> buffer)
	{
		buffer_ = buffer;
		bufferSet_ = true;
	}

	bool isBufferSet() { return bufferSet_; }

	/**
	 * @brief Opens stream and pushes events to Buffer
	 * @details [long description]
	 */
	void startPublishing()
	{
		if(!openStream()) return;

        if (!eventPublisher_)
		{
			running_ = true;
			eventPublisher_ = std::make_shared<std::thread>(&EventReader::pollEventStream, this);
        }
	}

	void stopPublishing()
	{
		if(running_)
		{
			running_ = false;
			eventPublisher_->join();
		}
	}

	bool isPublishing() { return running_; }

private:
	/**
	 * @brief Opens stream and returns a bool if successful.
	 */
	bool openStream()
	{
		std::shared_ptr<Edvs::IEventStream> stream_ = Edvs::OpenEventStream(uri_);
		return stream_->is_open();
	}

	void pollEventStream()
	{
	    // capture events (run until end of file or Ctrl+C)
		while(!stream_->eos() && running_ == true)
		{
			// read events from stream
			auto events = stream_->read();
			// display message
			if(!events.empty())
			{
				// TODO PUSH TO BUFFER
			}
		}
	}


	bool bufferSet_;
	std::shared_ptr<Buffer> buffer_;

	std::string uri_;

	std::shared_ptr<Edvs::IEventStream> stream_;
	std::shared_ptr<std::thread> eventPublisher_;
	bool running_;
	
};

#endif //EVENT_READER_H