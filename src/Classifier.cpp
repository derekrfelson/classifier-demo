/*
 * Classifier.cpp
 *
 *  Created on: Mar 19, 2016
 *      Author: derek
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include "Dataset.h"
#include "Classifier.h"
#include "Partition.h"

using Decimal = Dataset::Decimal;

Classifier::Classifier(
		const std::vector<Dataset::CovarianceMatrix>& cmInverses,
		const std::vector<Dataset::Decimal>& cmDeterminants,
		const std::vector<Dataset::RowVector>& meanVectors)
: cmInverses{cmInverses},
  cmDeterminants{cmDeterminants},
  meanVectors{meanVectors}
{
}

uint8_t Classifier::classify(Dataset::RowVector point) const
{
	for (auto a = 0; a < meanVectors.size(); ++a)
	{
		auto allPositive = true;

		// Compare A to everything else too see if anything is better
		for (auto b = 0; b < meanVectors.size(); ++b)
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
