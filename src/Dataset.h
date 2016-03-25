/*
 * Dataset.h
 *
 *  Created on: Mar 21, 2016
 *      Author: derek
 */

#ifndef DATASET_H_
#define DATASET_H_

#include "Classifier.h"
#include "Partition.h"
#include "Types.h"
#include <string>
#include <memory>
#include <vector>

class BayesClassifier;

class Dataset
{
public:
	const size_t NumFields;
	const size_t NumClasses;

public:
	explicit Dataset(std::vector<std::string> names,
				TypeVector types, DataMatrix data,
				size_t numClasses);
	size_t size() const;
	RowVector getMeans() const;
	Dataset getSubsetByClass(uint8_t type) const;
	Partition<Dataset> partition(size_t startIndex, size_t endIndex) const;
	RowVector getPoint(size_t i) const;
	uint8_t getType(size_t i) const;
	std::string getName(size_t i) const;
	CovarianceMatrix getCovarianceMatrix(ClassifierType type) const;
	BayesClassifier classifier(ClassifierType type) const;
	DataMatrix getData() const;
	void shuffle();

private:
	std::vector<std::string> names;
	TypeVector types;
	DataMatrix data;
};

CovarianceMatrix getPseudoInverse(const CovarianceMatrix& matrix);
Decimal getPseudoDeterminant(const CovarianceMatrix& matrix);

#endif /* DATASET_H_ */
