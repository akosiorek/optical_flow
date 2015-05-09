//
// Created by David Adrian

#include "EventReader.h"

void EventReader::openURIStream(const std::string& uri)
{
	stream = Edvs::OpenEventStream(uri);
}