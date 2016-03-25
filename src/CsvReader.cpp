/*
 * CsvReader.cpp
 *
 *  Created on: Mar 25, 2016
 *      Author: derek
 */

#include "CsvReader.h"
#include "Dataset.h"
#include <fstream>

static Dataset readDataset(std::string filename,
		size_t numFields,
		size_t numClasses,
		std::function<std::string(std::stringstream&)> nameReader,
		std::function<uint8_t(std::stringstream&)> typeReader)
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
	TypeVector types(numLines, 1);
	DataMatrix data(numLines, numFields);
	std::vector<std::string> names{};
	names.reserve(numLines);

	for (auto i = 0; i < numLines; ++i)
	{
		// Read the line
		std::getline(file, line);

		// Local variables
		auto ssLine = std::stringstream{line};
		auto field = std::string{};

		// Read the name
		names.push_back(nameReader(ssLine));

		// Read the data fields
		for (auto j = 0; j < numFields; ++j)
		{
			assert(std::getline(ssLine, field, ','));
			data(i, j) = std::stod(field);
		}

		// Read the class/type
		types[i] = typeReader(ssLine);

		// Must be at end of string now
		assert(!std::getline(ssLine, field, ','));
	}

	// Sanity check
	assert(names.size() == data.rows() && data.rows() == types.rows());

	// We must be at the end of the file now
	assert(!std::getline(file, line));

	return Dataset{names, types, data, numClasses};
}

// Iris data format has no name field
auto irisNameReader = [](std::stringstream& ssLine) {
	return "iris";
};

// Wine has no name
auto wineNameReader = [](std::stringstream& ssLine) {
	return "wine";
};

// Heart Disease data format has no name
auto heartDiseaseNameReader = [](std::stringstream& ssLine) {
	return "heartdisease";
};

// Iris stores type as one of 3 possible strings
auto irisTypeReader = [](std::stringstream& ssLine) {
	std::string field;
	std::getline(ssLine, field, ',');
	uint8_t ret = 0;

	if (field.compare(0, 11, "Iris-setosa") == 0)
	{
		ret = 1;
	}
	else if (field.compare(0, 15, "Iris-versicolor") == 0)
	{
		ret = 2;
	}
	else
	{
		ret = 3;
	}
	return ret;
};

// Heart Disease data format stores type-1 instead of type
auto heartDiseaseTypeReader = [](std::stringstream& ssLine) {
	std::string field;
	std::getline(ssLine, field, ',');
	return std::stoi(field) + 1;
};

// Wine data format stores type as an integer
auto wineTypeReader = [](std::stringstream& ssLine) {
	std::string field;
	std::getline(ssLine, field, ',');
	return std::stoi(field);
};

Dataset readIrisDataset(std::string filename)
{
	return readDataset(filename, IrisFields, IrisClasses,
			irisNameReader, irisTypeReader);
}

Dataset readWineDataset(std::string filename)
{
	return readDataset(filename, WineFields, WineClasses,
			wineNameReader, wineTypeReader);
}

Dataset readHeartDiseaseDataset(std::string filename)
{
	return readDataset(filename, HeartDiseaseFields, HeartDiseaseClasses,
			heartDiseaseNameReader, heartDiseaseTypeReader);
}
