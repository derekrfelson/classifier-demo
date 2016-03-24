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
#include <cassert>

using DataMatrix = Dataset::DataMatrix;
using TypeVector = Dataset::TypeVector;

/**
 * Calculates the entropy of a dataset by splitting it into two parts
 * based on class and seeing what proportion of the data belongs to each.
 */
double entropy(const Dataset::TypeVector& types)
{
	assert(types.rows() > 0);

	std::vector<uint8_t> vec(types.data(), types.data() + types.rows());
	std::sort(begin(vec), end(vec));

	double ret = 0;
	auto currentType = vec[0];
	auto currentTypeMatches = 0;
	for (auto type : vec)
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
