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
#include <eigen3/Eigen/Dense>
#include "Dataset.h"

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
			const std::vector<Dataset::CovarianceMatrix>& cmInverses,
			const std::vector<Dataset::Decimal>& cmDeterminants,
			const std::vector<Dataset::RowVector>& meanVectors);
	uint8_t classify(Dataset::RowVector point) const;

private:
	std::vector<Dataset::CovarianceMatrix> cmInverses;
	std::vector<Dataset::Decimal> cmDeterminants;
	std::vector<Dataset::RowVector> meanVectors;
};

#endif /* CLASSIFIER_H_ */
