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
  type{0},
  data{0},
  means{0},
  covarianceMatrix{ AnimalMatrix::Zero() }
{
	auto ssLine = std::stringstream{csvLine};
	auto field = std::string{};

	// Read the name
	assert(std::getline(ssLine, field, ','));
	name = field;

	// Read the 16 data fields
	for (auto i = 0; i < 16; ++i)
	{
		assert(std::getline(ssLine, field, ','));
		data[i] = std::stoi(field);
	}

	// Read the class/type
	assert(std::getline(ssLine, field, ','));
	type = std::stoi(field);

	// Must be at end of string now
	assert(!std::getline(ssLine, field, ','));

	// Certain fields can only take on certain values
	assert(data[12] == 0 || data[12] == 2 || data[12] == 4
			|| data[12] || data[12] == 5 || data[12] == 6
			|| data[12] == 8);
	assert(data[16] >= 1 && data[16] <= 7);
}

uint8_t Animal::getType() const
{
	return type;
}
