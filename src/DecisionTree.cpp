/*
 * DecisionTree.cpp
 *
 *  Created on: Mar 24, 2016
 *      Author: derek
 */

#include "DecisionTree.h"
#include "Dataset.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>

using DataMatrix = Dataset::DataMatrix;
using TypeVector = Dataset::TypeVector;
using ColVector = Dataset::ColVector;

/**
 * Calculates the entropy of a dataset by splitting it into two parts
 * based on class and seeing what proportion of the data belongs to each.
 */
double entropy(const TypeVector& types)
{
	assert(types.rows() > 0);

	std::vector<uint8_t> sortedTypes(types.data(),
			types.data() + types.rows());
	std::sort(begin(sortedTypes), end(sortedTypes));

	double ret = 0;
	auto currentType = sortedTypes[0];
	auto currentTypeMatches = 0;
	for (auto type : sortedTypes)
	{
		if (currentType == type)
		{
			++currentTypeMatches;
		}
		else
		{
			// Every time we see a new class, add the entropy from the last one
			auto p = static_cast<double>(currentTypeMatches) / types.rows();
			ret -= (p) * log2(p);
			currentTypeMatches = 1;
			currentType = type;
		}
	}

	// Add the entropy from the final class
	auto p = static_cast<double>(currentTypeMatches) / types.rows();
	ret -= p * log2(p);

	return ret;
}

double entropy(const std::vector<uint8_t>& types)
{
	TypeVector tv(types.size());
	for (auto i = 0; i < types.size(); ++i)
	{
		tv[i] = types[i];
	}
	return entropy(tv);
}

/*
 * Calculates how many bits you will save by knowing the value of an attribute.
 * It's a measure of how well the given column of your data can predict
 * the type.
 */
double gain(const TypeVector& types, const ColVector& dataColumn)
{
	assert(types.rows() > 0);
	assert(types.rows() == dataColumn.rows());

	// Find all the unique values in the column
	std::vector<uint8_t> uniqueValues(dataColumn.data(),
			dataColumn.data() + dataColumn.rows());
	std::sort(begin(uniqueValues), end(uniqueValues));
	uniqueValues.erase(std::unique(begin(uniqueValues),
			end(uniqueValues)), end(uniqueValues));

	// Gain is entropy(types) - something per each unique value
	double ret = entropy(types);

	for (auto value : uniqueValues)
	{
		// Select all the data points where that column = that value
		std::vector<uint8_t> subsetTypes;
		for (auto i = 0; i < dataColumn.rows(); ++i)
		{
			if (dataColumn[i] == value)
			{
				subsetTypes.push_back(types[i]);
			}
		}

		assert(subsetTypes.size() > 0);

		// Add the entropy gained by knowing that column = that value
		ret -= static_cast<double>(subsetTypes.size())/dataColumn.rows()
				* entropy(subsetTypes);
	}

	return ret;
}

/*
 * Gives you the 0-based index of the column that maximizes information gain.
 */
size_t bestAttribute(const Dataset::TypeVector& types,
		const Dataset::DataMatrix& data)
{
	double maxGain = -999;
	size_t ret = 999;
	for (auto i = 0; i < data.cols(); ++i)
	{
		auto colGain = gain(types, data.col(i));
		if (colGain > maxGain)
		{
			ret = i;
			maxGain = colGain;
		}
	}
	return ret;
}
