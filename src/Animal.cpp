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
  hair{false},
  feathers{false},
  eggs{false},
  milk{false},
  airborne{false},
  aquatic{false},
  predator{false},
  toothed{false},
  backbone{false},
  breathes{false},
  venomous{false},
  fins{false},
  legs{0},
  tail{false},
  domestic{false},
  catsize{false},
  type{0}
{
	auto field = std::string{};
	auto ssLine = std::stringstream{csvLine};
	auto ssField = std::stringstream{};

	assert(std::getline(ssLine, field, ','));
	name = field;

	assert(std::getline(ssLine, field, ','));
	hair = field == "1";

	assert(std::getline(ssLine, field, ','));
	feathers = field == "1";

	assert(std::getline(ssLine, field, ','));
	eggs = field == "1";

	assert(std::getline(ssLine, field, ','));
	milk = field == "1";

	assert(std::getline(ssLine, field, ','));
	airborne = field == "1";

	assert(std::getline(ssLine, field, ','));
	aquatic = field == "1";

	assert(std::getline(ssLine, field, ','));
	predator = field == "1";

	assert(std::getline(ssLine, field, ','));
	toothed = field == "1";

	assert(std::getline(ssLine, field, ','));
	backbone = field == "1";

	assert(std::getline(ssLine, field, ','));
	breathes = field == "1";

	assert(std::getline(ssLine, field, ','));
	venomous = field == "1";

	assert(std::getline(ssLine, field, ','));
	fins = field == "1";

	assert(std::getline(ssLine, field, ','));
	ssField = std::stringstream{field};
	ssField >> legs;
	assert(legs == 0 || legs == 2 || legs == 4 || legs == 5 || legs == 6
			|| legs == 8);

	assert(std::getline(ssLine, field, ','));
	tail = field == "1";

	assert(std::getline(ssLine, field, ','));
	domestic = field == "1";

	assert(std::getline(ssLine, field, ','));
	catsize = field == "1";

	assert(std::getline(ssLine, field, ','));
	ssField = std::stringstream{field};
	ssField >> type;

	// Must be at end of string now
	assert(!std::getline(ssLine, field, ','));
}
