/*
 * Classifier.h
 *
 *  Created on: Mar 19, 2016
 *      Author: derek
 */

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include <vector>
#include <cstdint>
#include "ZooDataset.h"

enum class ClassifierType : uint8_t
{
	OPTIMAL,
	NAIVE,
	LINEAR
};

struct Classifier
{
public:
	explicit Classifier(
			const std::vector<ZooDataset::CovarianceMatrix>& cmInverses,
			const std::vector<ZooDataset::Decimal>& cmDeterminants,
			const std::vector<ZooDataset::RowVector>& meanVectors);
	uint8_t classify(ZooDataset::RowVector point) const;

private:
	std::vector<ZooDataset::CovarianceMatrix> cmInverses;
	std::vector<ZooDataset::Decimal> cmDeterminants;
	std::vector<ZooDataset::RowVector> meanVectors;
};

#endif /* CLASSIFIER_H_ */
