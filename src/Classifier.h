/*
 * Classifier.h
 *
 *  Created on: Mar 19, 2016
 *      Author: derek
 */

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include "Types.h"
#include <cstdint>

class Classifier
{
public:
	virtual uint8_t classify(const RowVector& point) const = 0;
	virtual ~Classifier() = default;
};

#endif /* CLASSIFIER_H_ */
