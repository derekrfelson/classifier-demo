/*
 * Partition.cpp
 *
 *  Created on: Mar 18, 2016
 *      Author: derek
 */

#include <cassert>
#include "Partition.h"

/**
 * Computes start and end indices for slicing a dataset up for k-fold
 * cross-validation.
 *
 * currentFold: Which cross-validation iteration we're on. Starts at 1.
 *
 * k: How many times we want to do cross validation before every
 *    element has been tested.
 */
std::pair<size_t, size_t> kFoldIndices(size_t currentFold, size_t k, size_t size)
{
	// Check preconditions
	assert(currentFold > 0 && currentFold <= k);
	assert(size > 1);
	assert(k <= size && k > 1);

	// Calculate start index
	auto numTestingElements = static_cast<size_t>(size / k);
	auto startIndex = numTestingElements * (currentFold - 1);

	// Calculate end index
	auto endIndex = startIndex + numTestingElements - 1;

	// Sometimes the dataset will be the wrong size to split into k parts.
	// Just to be safe, on the last iteration we make sure we put all the
	// remaining elements into the testing set.
	if (currentFold == k)
	{
		endIndex = size - 1;
	}

	return {startIndex, endIndex};
}
