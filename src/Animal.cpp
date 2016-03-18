/*
 * Animal.cpp
 *
 *  Created on: Mar 17, 2016
 *      Author: derek
 */

#include "Animal.h"
#include <cassert>
#include <sstream>
#include <iostream>

Animal::Animal(std::string csvLine)
: name{},
  fields{}
{
	auto ssLine = std::stringstream{csvLine};
	auto field = std::string{};

	// Read the name
	assert(std::getline(ssLine, field, ','));
	name = field;

	// Read the 17 remaining fields
	for (auto i = 0; i < 17; ++i)
	{
		assert(std::getline(ssLine, field, ','));
		fields[i] = std::stoi(field);
	}

	// Must be at end of string now
	assert(!std::getline(ssLine, field, ','));

	// Certain fields can only take on certain values
	assert(fields[12] == 0 || fields[12] == 2 || fields[12] == 4
			|| fields[12] || fields[12] == 5 || fields[12] == 6
			|| fields[12] == 8);
	assert(fields[16] >= 1 && fields[16] <= 7);
}

uint8_t Animal::getType() const
{
	return fields[16];
}
