/*
 * DecisionTree.h
 *
 *  Created on: Mar 24, 2016
 *      Author: derek
 */

#ifndef DECISIONTREE_H_
#define DECISIONTREE_H_

#include "Classifier.h"
#include <cstdint>
#include <list>
#include <memory>
#include <iosfwd>
#include <string>

class DecisionTree : public Classifier
{
public:
	explicit DecisionTree(const TypeVector& types,
			const DataMatrix& data);
	uint8_t classify(const RowVector& dataPoint) const override;
	std::ostream& print(std::ostream& out) const;

private:
	struct Node
	{
	public:
		explicit Node(size_t attributeIndex, const TypeVector& types,
				const DataMatrix& data);
		explicit Node(size_t attributeIndex, uint8_t parentType,
				const Node* parent, const TypeVector& types,
				const DataMatrix& data, size_t attributesChecked);
		std::ostream& print(std::ostream& out) const;
		std::string name() const;

		size_t attributeIndex;
		uint8_t parentAttrValue;
		std::list<Node> children;
		const Node* parent;
		uint8_t type;
	};
	friend Node;

	std::unique_ptr<Node> root;
};

double entropy(const TypeVector& types);
double entropy(const std::vector<uint8_t>& types);
double gain(const TypeVector& types,
		const ColVector& dataColumn);
size_t bestAttribute(const TypeVector& types,
		const DataMatrix& data);
std::ostream& operator<<(std::ostream& out, const DecisionTree& dt);

#endif /* DECISIONTREE_H_ */
