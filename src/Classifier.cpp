/*
 * Classifier.cpp
 *
 *  Created on: Mar 19, 2016
 *      Author: derek
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include "ZooDataset.h"
#include "Classifier.h"
#include "Partition.h"

using Decimal = ZooDataset::Decimal;

Classifier::Classifier(const ZooDataset& trainingSet)
: cmInverses{},
  cmDeterminants{},
  meanVectors{}
{
	cmInverses.reserve(ZooDataset::NumClasses);
	cmDeterminants.reserve(ZooDataset::NumClasses);
	meanVectors.reserve(ZooDataset::NumClasses);

	// Split the training data into classes
	for (auto i = 1; i <= ZooDataset::NumClasses; ++i)
	{
		auto trainingClass = trainingSet.getSubsetByClass(i);
		cmInverses.push_back(
				trainingClass.getCovarianceMatrixInverse());
		cmDeterminants.push_back(
				trainingClass.getCovarianceMatrixDeterminant());
		meanVectors.push_back(
				trainingClass.getMeans());
	}
}

uint8_t Classifier::classify(ZooDataset::RowVector point) const
{
	for (auto a = 0; a < ZooDataset::NumClasses; ++a)
	{
		auto allPositive = true;

		// Compare A to everything else too see if anything is better
		for (auto b = 0; b < ZooDataset::NumClasses; ++b)
		{
			if (a == b)
			{
				continue;
			}

			auto value =
					std::log(cmDeterminants[b]) - std::log(cmDeterminants[a])
				    + (point - meanVectors[b])
					   * cmInverses[b]
					   * (point - meanVectors[b]).transpose()
			        - (point - meanVectors[a])
					   * cmInverses[a]
					   * (point - meanVectors[a]).transpose();

			if (value < 0)
			{
				// The point was closer to the other mean, so reject class A
				allPositive = false;
				break;
			}
		}

		// If we didn't find anything better, A really is our best class
		if (allPositive)
		{
			return a + 1; // Types start at index 1, but vectors at index 0
		}
	}

	assert(false); // Failed to find a class the point was closest to
}
