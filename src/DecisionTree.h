/*
 * DecisionTree.h
 *
 *  Created on: Mar 24, 2016
 *      Author: derek
 */

#ifndef DECISIONTREE_H_
#define DECISIONTREE_H_

#include "Dataset.h"
#include <cstdint>

double entropy(const Dataset::TypeVector& types);
double entropy(const std::vector<uint8_t>& types);
double gain(const Dataset::TypeVector& types,
		const Dataset::ColVector& dataColumn);
size_t bestAttribute(const Dataset::TypeVector& types,
		const Dataset::DataMatrix& data);

#endif /* DECISIONTREE_H_ */
