/*
 * Dataset.cpp
 *
 *  Created on: Mar 18, 2016
 *      Author: derek
 */

#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <eigen3/Eigen/SVD>
#include "Dataset.h"
#include "Classifier.h"

using Decimal = Dataset::Decimal;
using DataMatrix = Dataset::DataMatrix;
using RowVector = Dataset::RowVector;
using TypeVector = Dataset::TypeVector;
using CovarianceMatrix = Dataset::CovarianceMatrix;

Dataset::Dataset(std::vector<std::string> names, TypeVector types,
		DataMatrix data, size_t numClasses)
: NumFields{static_cast<size_t>(data.cols())},
  NumClasses{numClasses},
  names{names},
  types{types},
  data{data}
{
}

size_t Dataset::size() const
{
	return names.size();
}

/**
 * Separates the dataset into training and testing sets. The result will
 * be a partition where testing = elements in [startIndex, endIndex],
 * and training = everything else.
 */
Partition<Dataset> Dataset::partition(
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

	return Partition<Dataset>{
		Dataset{trainingNames, trainingTypes, trainingData, NumClasses},
		Dataset{testingNames, testingTypes, testingData, NumClasses}};
}

Dataset Dataset::getSubsetByClass(uint8_t type) const
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
		std::cerr << "Problem found: Dataset subclass of type "
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

	return Dataset{std::move(subsetNames), std::move(subsetTypes),
		std::move(subsetData), NumClasses};
}

RowVector Dataset::getMeans() const
{
	return data.colwise().mean();
}

CovarianceMatrix Dataset::getCovarianceMatrix(ClassifierType type) const
{
	if (type == ClassifierType::LINEAR)
	{
		// Linear Bayes classifier assumes that all dimensions
		// are independent and have the same variance, so our
		// covariance matrix is just the identity matrix.
		return CovarianceMatrix::Identity(NumFields, NumFields);
	}

	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> centered
		= data.rowwise() - getMeans();
	CovarianceMatrix cov = (centered.transpose() * centered)
			/ static_cast<Decimal>(data.rows() - 1);

	// Return the correct type of covariance matrix
	if (type == ClassifierType::OPTIMAL)
	{
		// Optimal Bayes classifier uses the full covariance matrix.
		return cov;
	}
	else
	{
		// Naive Bayes classifier assumes all dimensions independent,
		// so we set all the non-diagonal entries in the covariance
		// matrix to zero.
		return cov.diagonal().asDiagonal();
	}
}

Dataset::RowVector Dataset::getPoint(size_t i) const
{
	assert(i < data.rows());
	return data.row(i);
}

uint8_t Dataset::getType(size_t i) const
{
	assert(i < types.rows());
	return types[i];
}

std::string Dataset::getName(size_t i) const
{
	assert(i < names.size());
	return names[i];
}

Classifier Dataset::classifier(ClassifierType type) const
{
	std::vector<CovarianceMatrix> cmInverses;
	std::vector<Decimal> cmDeterminants;
	std::vector<RowVector> meanVectors;
	cmInverses.reserve(NumClasses);
	cmDeterminants.reserve(NumClasses);
	meanVectors.reserve(NumClasses);

	// Split the training data into classes
	for (auto i = 1; i <= NumClasses; ++i)
	{
		auto trainingClass = getSubsetByClass(i);
		auto cv = trainingClass.getCovarianceMatrix(type);
		cmInverses.push_back(getPseudoInverse(cv));
		cmDeterminants.push_back(getPseudoDeterminant(cv));
		meanVectors.push_back(trainingClass.getMeans());
	}

	return Classifier{cmInverses, cmDeterminants, meanVectors};
}

Dataset::CovarianceMatrix getPseudoInverse(
		const Dataset::CovarianceMatrix& matrix)
{
	auto svd = matrix.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

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

	auto pinvM = static_cast<Dataset::CovarianceMatrix>(
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

Decimal getPseudoDeterminant(
		const Dataset::CovarianceMatrix& matrix)
{
	auto svs = matrix.jacobiSvd().singularValues();
	assert(svs.rows() == 16);

	Decimal product = 1;
	for (auto i = 0; i < svs.rows(); ++i)
	{
		if (svs[i] > .00000001)
		{
			product *= svs[i];
		}
	}

	return product;
}
