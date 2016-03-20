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

struct Classifier
{
public:
	explicit Classifier(const ZooDataset& trainingSet);
	uint8_t classify(ZooDataset::RowVector point) const;

private:
	std::vector<ZooDataset::CovarianceMatrix> cmInverses;
	std::vector<double> cmDeterminants;
	std::vector<ZooDataset::RowVector> meanVectors;
};

#endif /* CLASSIFIER_H_ */
