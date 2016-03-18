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
	data = DataMatrix(numLines, NumFields);
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
}

size_t ZooDataset::size() const
{
	return names.size();
}

Partition<ZooDataset> ZooDataset::partition(size_t startingFold,
		size_t numFolds)
{
}

/**
 * Separates the dataset into training and testing sets using
 * the leave-one-out method. The result will be two datasets,
 * where one contains a single element and the other contains
 * the remainder of the elements.
 *
 * leaveOutIndex: the index of the element to leave out
 */
Partition<ZooDataset> ZooDataset::partition(size_t leaveOutIndex)
{
	assert(leaveOutIndex >= 0 && leaveOutIndex < size());
	assert(size() >= 2);

	// Initialize the sizes for the training subset
	std::vector<std::string> trainingNames{};
	trainingNames.reserve(size()-1);
	TypeVector trainingTypes{size()-1, 1};
	DataMatrix trainingData{size()-1, NumFields};

	// Initialize the sizes for the test subset (all 1)
	TypeVector testingTypes{1, 1};
	DataMatrix testingData{1, NumFields};
	std::vector<std::string> testingNames{};

	// Fill out the test subset now
	testingTypes[0] = types[leaveOutIndex];
	testingData.row(0) = data.row(leaveOutIndex);
	testingNames.push_back(names[leaveOutIndex]);

	// Fill out the training subset with the remaining elements
	auto trainingSize = 0;
	for (auto i = 0; i < size(); ++i)
	{
		if (i != leaveOutIndex)
		{
			trainingNames.push_back(names[i]);
			trainingTypes[trainingSize] = types[i];
			trainingData.row(trainingSize) = data.row(i);
			++trainingSize;
		}
	}

	// Sanity check
	assert(trainingSize == size() - 1);
	assert(trainingNames.size() == trainingSize);
	assert(trainingData.rows() == trainingSize);
	assert(trainingTypes.rows() == trainingSize);
	assert(testingNames.size() == 1);
	assert(testingData.rows() == 1);
	assert(testingTypes.rows() == 1);

	return Partition<ZooDataset>{
		ZooDataset{trainingNames, trainingTypes, trainingData},
		ZooDataset{testingNames, testingTypes, testingData}};
}

ZooDataset::ZooDataset(std::vector<std::string> names, TypeVector types,
		DataMatrix data)
: names{names},
  types{types},
  data{data}
{
}

ZooDataset ZooDataset::getSubsetByClass(uint8_t type) const
{
	assert(type >= 1 && type <= NumClasses);

	// Count how large the subset will be
	auto subsetSize = 0;
	for (auto i = 0; i < names.size(); ++i)
	{
		if (types[i] == type)
		{
			++subsetSize;
		}
	}

	// End the program if we have a size-0 subclass
	if (subsetSize == 0)
	{
		std::cerr << "Problem found: ZooDataset subclass of type "
				<< static_cast<int>(type) << " has size 0" << std::endl;
		std::abort();
	}

	// Initialize our vectors to the right size
	// Subset types will all be the same
	auto subsetTypes = TypeVector::Constant(subsetSize, 1, type);
	auto subsetData = DataMatrix(subsetSize, NumFields);
	auto subsetNames = std::vector<std::string>(subsetSize);

	// Copy the data into the subset
	auto numCopied = 0;
	for (auto i = 0; i < names.size(); ++i)
	{
		if (types[i] == type)
		{
			subsetNames.push_back(names[i]);
			subsetData.row(numCopied++) = data.row(i);
		}
	}

	// Sanity check
	assert(numCopied == subsetSize);

	return ZooDataset{std::move(subsetNames), std::move(subsetTypes),
		std::move(subsetData)};
}

ZooDataset::MeanRowVector ZooDataset::getMeans() const
{
	return data.cast<double>().colwise().mean();
}

ZooDataset::CovarianceMatrix ZooDataset::getCovarianceMatrix() const
{
	Eigen::MatrixXd centered = data.cast<double>().rowwise() - getMeans();
	CovarianceMatrix cov = (centered.transpose() * centered)
			/ static_cast<double>(data.rows() - 1);
	return cov;
}


