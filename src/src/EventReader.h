#ifndef EVENT_READER_H
#define EVENT_READER_H

#include <string>

#include "Edvs/event.h"
#include "Edvs/EventStream.hpp"

/**
 * @brief Provide easy access to a stream of event
 * @details [long description]
 * @return [description]
 */
class EventReader
{
public:
	EventReader();
	~EventReader();

	void openURIStream(const std::string& uri);
	void startReadingEvents();

private:
	std::shared_ptr<Edvs::IEventStream> stream;
	
};

#endif //EVENT_READER_H
