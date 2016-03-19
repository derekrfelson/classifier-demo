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
#include <eigen3/Eigen/SVD>

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

/**
 * Separates the dataset into training and testing sets. The result will
 * be a partition where testing = elements in [startIndex, endIndex],
 * and training = everything else.
 */
Partition<ZooDataset> ZooDataset::partition(
		size_t startIndex, size_t endIndex) const
{
	// Check preconditions
	assert(startIndex <= endIndex);
	assert(endIndex < size());

	// Calculate how large each subset will be
	const auto finalTestingSize = endIndex - startIndex + 1;
	const auto finalTrainingSize = size() - finalTestingSize;

	// Initialize the sizes for the training subset
	std::vector<std::string> trainingNames{};
	trainingNames.reserve(finalTrainingSize);
	TypeVector trainingTypes{finalTrainingSize, 1};
	DataMatrix trainingData{finalTrainingSize, NumFields};

	// Initialize the sizes for the test subset
	std::vector<std::string> testingNames{};
	testingNames.reserve(finalTestingSize);
	TypeVector testingTypes{finalTestingSize, 1};
	DataMatrix testingData{finalTestingSize, NumFields};

	// Fill out the training subset with the remaining elements
	auto trainingSize = 0;
	auto testingSize = 0;
	for (auto i = 0; i < size(); ++i)
	{
		if (i >= startIndex && i <= endIndex)
		{
			testingNames.push_back(names[i]);
			testingTypes[testingSize] = types[i];
			testingData.row(testingSize) = data.row(i);
			++testingSize;
		}
		else
		{
			trainingNames.push_back(names[i]);
			trainingTypes[trainingSize] = types[i];
			trainingData.row(trainingSize) = data.row(i);
			++trainingSize;
		}
	}

	// Sanity check
	assert(trainingSize == finalTrainingSize);
	assert(trainingNames.size() == finalTrainingSize);
	assert(trainingData.rows() == finalTrainingSize);
	assert(trainingTypes.rows() == finalTrainingSize);
	assert(testingSize == finalTestingSize);
	assert(testingNames.size() == finalTestingSize);
	assert(testingData.rows() == finalTestingSize);
	assert(testingTypes.rows() == finalTestingSize);

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
	// Check preconditions
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

ZooDataset::CovarianceMatrix ZooDataset::getCovarianceMatrixInverse() const
{
	auto svd = getCovarianceMatrix().jacobiSvd(
			Eigen::ComputeFullU | Eigen::ComputeFullV);

	auto pinvS = svd.singularValues();
	for (auto i = 0; i < pinvS.rows(); ++i)
	{
		// Anything too small is probably a rounding error
		if (pinvS[i] < .00000001)
		{
			pinvS[i] = 0;
		}

		// Take the reciprocal
		if (pinvS[i] > 0)
		{
			pinvS[i] = 1 / pinvS[i];
		}
	}

	auto pinvM = static_cast<ZooDataset::CovarianceMatrix>(
			svd.matrixV() * pinvS.asDiagonal() * svd.matrixU().transpose());

	for (auto i = 0; i < pinvM.rows(); ++i)
	{
		for (auto j = 0; j < pinvM.cols(); ++j)
		{
			// Anything too small is probably a rounding error
			if (pinvM(i,j) < .00000001)
			{
				pinvM(i,j) = 0;
			}
		}
	}

	return pinvM;
}

double ZooDataset::getCovarianceMatrixDeterminant() const
{
	return getCovarianceMatrix().jacobiSvd().singularValues().sum();
}
