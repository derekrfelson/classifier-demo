/*
 * Classifier.h
 *
 *  Created on: Mar 19, 2016
 *      Author: derek
 */

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include "ZooDataset.h"

struct Classifier
{
public:
	explicit Classifier(const ZooDataset& dataset);

	ZooDataset::CovarianceMatrix cmInverse;
	double cmDeterminant;
	ZooDataset::MeanRowVector means;
};

#endif /* CLASSIFIER_H_ */
