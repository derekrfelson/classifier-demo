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
#include <list>
#include <memory>

class DecisionTree
{
public:
	explicit DecisionTree(const Dataset::TypeVector& types,
			const Dataset::DataMatrix& data);
	uint8_t classify(const Dataset::RowVector& dataPoint) const;

private:
	struct Node
	{
	public:
		explicit Node(size_t attributeIndex, const Dataset::TypeVector& types,
				const Dataset::DataMatrix& data);
		explicit Node(size_t attributeIndex, uint8_t parentType,
				const Node* parent, const Dataset::TypeVector& types,
				const Dataset::DataMatrix& data);

		size_t attributeIndex;
		uint8_t parentType;
		std::list<Node> children;
		const Node* parent;
		std::unique_ptr<Dataset::TypeVector> types;
		std::unique_ptr<Dataset::DataMatrix> data;
	};
	friend Node;

	std::unique_ptr<Node> root;
};

double entropy(const Dataset::TypeVector& types);
double entropy(const std::vector<uint8_t>& types);
double gain(const Dataset::TypeVector& types,
		const Dataset::ColVector& dataColumn);
size_t bestAttribute(const Dataset::TypeVector& types,
		const Dataset::DataMatrix& data);

#endif /* DECISIONTREE_H_ */
