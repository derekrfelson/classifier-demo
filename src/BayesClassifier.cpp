/*
 * BayesClassifier.cpp
 *
 *  Created on: Mar 25, 2016
 *      Author: derek
 */

#include "BayesClassifier.h"
#include <cmath>

BayesClassifier::BayesClassifier(
		const std::vector<CovarianceMatrix>& cmInverses,
		const std::vector<Decimal>& cmDeterminants,
		const std::vector<RowVector>& meanVectors)
: cmInverses{cmInverses},
  cmDeterminants{cmDeterminants},
  meanVectors{meanVectors}
{
}

uint8_t BayesClassifier::classify(const RowVector& point) const
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
	return 0;
}
