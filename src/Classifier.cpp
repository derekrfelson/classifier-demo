/*
 * Classifier.cpp
 *
 *  Created on: Mar 19, 2016
 *      Author: derek
 */

#include "Classifier.h"

Classifier::Classifier(const ZooDataset& dataset)
: cmInverse{dataset.getCovarianceMatrixInverse()},
  cmDeterminant{dataset.getCovarianceMatrixDeterminant()},
  means{dataset.getMeans()}
{
}
