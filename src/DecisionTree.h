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

class DecisionTree
{
};

double entropy(const Dataset::TypeVector& types);
double entropy(const Dataset::TypeVector& types, uint8_t positiveType);


#endif /* DECISIONTREE_H_ */
