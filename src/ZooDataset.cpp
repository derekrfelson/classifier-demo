/*
 * ZooDataset.cpp
 *
 *  Created on: Mar 18, 2016
 *      Author: derek
 */

#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <eigen3/Eigen/SVD>
#include "Dataset.h"

using Decimal = Dataset::Decimal;

constexpr auto NumFields = 16;
constexpr auto NumClasses = 7;

Dataset readZooDataset(std::string filename)
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
	Dataset::TypeVector types(numLines, 1);
	Dataset::DataMatrix data(numLines, NumFields);
	std::vector<std::string> names{};
	names.reserve(numLines);

	for (auto i = 0; i < numLines; ++i)
	{
		// Read the line
		assert(std::getline(file, line));

		// Read the name
		auto ssLine = std::stringstream{line};
		auto field = std::string{};
		assert(std::getline(ssLine, field, ','));
		names.push_back(field);

		// Read the data fields
		for (auto j = 0; j < NumFields; ++j)
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
		assert(types[i] >= 1 && types[i] <= NumClasses);
	}

	// Sanity check
	assert(names.size() == data.rows() && data.rows() == types.rows());

	// We must be at the end of the file now
	assert(!std::getline(file, line));

	return Dataset{names, types, data, NumClasses};
}

