/*
 * BayesClassifier.h
 *
 *  Created on: Mar 25, 2016
 *      Author: derek
 */

#ifndef BAYESCLASSIFIER_H_
#define BAYESCLASSIFIER_H_

#include "Classifier.h"
#include "Types.h"
#include <vector>

class BayesClassifier: public Classifier
{
public:
	explicit BayesClassifier(
		const std::vector<CovarianceMatrix>& cmInverses,
		const std::vector<Decimal>& cmDeterminants,
		const std::vector<RowVector>& meanVectors);
	virtual uint8_t classify(const RowVector& point) const override;

private:
	std::vector<CovarianceMatrix> cmInverses;
	std::vector<Decimal> cmDeterminants;
	std::vector<RowVector> meanVectors;
};

#endif /* BAYESCLASSIFIER_H_ */
