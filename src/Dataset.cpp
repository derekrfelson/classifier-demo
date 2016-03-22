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

RowVector Dataset::getMeans() const
{
	return data.colwise().mean();
}

size_t Dataset::size() const
{
	return names.size();
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

DataMatrix Dataset::getData() const
{
	return data;
}

CovarianceMatrix getPseudoInverse(const CovarianceMatrix& matrix)
{
	if (matrix.determinant() != 0)
	{
		return matrix.inverse();
	}

	std::cout << "Using pseudoinverse!" << std::endl;

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
	if (matrix.determinant() != 0)
	{
		return matrix.determinant();
	}

	auto svs = matrix.jacobiSvd().singularValues();

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

Dataset readDataset(std::string filename,
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
	Dataset::TypeVector types(numLines, 1);
	Dataset::DataMatrix data(numLines, numFields);
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
