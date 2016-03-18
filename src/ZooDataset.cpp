/*
 * ZooDataset.cpp
 *
 *  Created on: Mar 18, 2016
 *      Author: derek
 */

#include "ZooDataset.h"
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <iostream>

ZooDataset::ZooDataset(std::string filename)
: names{},
  types{},
  data{}
{
	auto line = std::string{};

	// Figure out the size of the matrix and vectors we'll need
	auto file = std::ifstream{filename};
	auto numLines = 0;
	while(std::getline(file, line))
	{
		++numLines;
	}

	// Return to the start of the file
	file.clear();
	file.seekg(0, std::ios::beg);

	// Initialize our vectors to the right size
	types = TypeVector(numLines, 1);
	data = DataMatrix(numLines, 16);
	names.reserve(numLines);

	for (auto i = 0; i < numLines; ++i)
	{
		// Read the line
		assert(std::getline(file, line));

		// Read the name
		auto ssLine = std::stringstream{line};
		auto field = std::string{};
		assert(std::getline(ssLine, field, ','));
		names[i] = field;

		// Read the 16 data fields
		for (auto j = 0; j < 16; ++j)
		{
			assert(std::getline(ssLine, field, ','));
			data(i, j) = std::stoi(field);
		}

		// Read the class/type
		assert(std::getline(ssLine, field, ','));
		types[i] = std::stoi(field);

		// Must be at end of string now
		assert(!std::getline(ssLine, field, ','));

		// Certain fields can only take on certain values
		assert(data(i,12) == 0 || data(i,12) == 2 || data(i,12) == 4
				|| data(i,12) || data(i,12) == 5 || data(i,12) == 6
				|| data(i,12) == 8);
		assert(types[i] >= 1 && types[i] <= 7);
	}

	// We must be at the end of the file now
	assert(!std::getline(file, line));
}

