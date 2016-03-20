/*
 * Classifier.cpp
 *
 *  Created on: Mar 19, 2016
 *      Author: derek
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include "Classifier.h"
#include "Partition.h"

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
	auto bestType = 0;
	double bestValue = 0;

	for (auto a = 0; a < ZooDataset::NumClasses - 1; ++a)
	{
		// Assume that A is our best class
		bestType = a;

		// Compare A to everything else too see if anything is better
		for (auto b = a + 1; b < ZooDataset::NumClasses; ++b)
		{
			auto value =
					std::log(cmDeterminants[b]) - std::log(cmDeterminants[a])
				    + (point - meanVectors[b])
					   * cmInverses[b]
					   * (point - meanVectors[b]).transpose()
			        - (point - meanVectors[a])
					   * cmInverses[a]
					   * (point - meanVectors[a]).transpose();

			std::cout << "  Comparing " << (a+1) << " to " << (b+1) << " = " << value << std::endl;

			if (value < 0) // point is further from a's mean than b's mean
			{
				bestType = b;
				//break;
			}
		}

		// If we didn't find anything better, A really is our best class
		if (bestType == a)
		{
			std::cout << "  Good class was " << static_cast<int>(a+1) << std::endl;
			//break;
		}
	}

	return bestType + 1; // Types start at index 1, but vectors at index 0
}
