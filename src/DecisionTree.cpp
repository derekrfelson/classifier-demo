/*
 * DecisionTree.cpp
 *
 *  Created on: Mar 24, 2016
 *      Author: derek
 */

#include "DecisionTree.h"
#include "Types.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>
#include <ostream>
#include <iostream>
#include <sstream>

constexpr uint8_t NoType = 255;
constexpr size_t NoAttrIndex = 99999;
constexpr size_t NoParentAttrValue = 99999;

static std::vector<uint8_t> colToStdVector(const ColVector& col);
static std::vector<uint8_t> rowToStdVector(const RowVector& row);
static std::vector<uint8_t> getUniqueValues(std::vector<uint8_t> vec);

DecisionTree::DecisionTree(const TypeVector& types, const DataMatrix& data)
: nodeCount{0}
{
	root = std::make_unique<Node>(types, data, *this);
}

/**
 * Initialize a root node on the decision tree.
 */
DecisionTree::Node::Node(const TypeVector& types, const DataMatrix& data,
		DecisionTree& dt)
: Node{NoParentAttrValue, nullptr, types, data, 0, dt}
{
}

/**
 * Initialize a child node on the decision tree.
 *
 * Each node on the tree looks at an attribute, the value of which can
 * fall into several different classes.
 *
 * parentAttrValue: Value matched for the parent's distinguishing attribute
 */
DecisionTree::Node::Node(size_t parentAttrValue, const Node* parent,
		const TypeVector& types, const DataMatrix& data,
		 size_t attributesChecked, DecisionTree& dt)
: parentAttrValue{parentAttrValue},
  parent{parent},
  attributesChecked{attributesChecked},
  children{},
  attributeIndex{NoAttrIndex},
  type{NoType},
  nodeNumber{dt.nodeCount++}
{
	// Turn this node into a correct leaf node, or make its children
	if (entropy(types) == 0)
	{
		// Case 1: The subset consists of only 1 type. Perfect!

		// In an ideal leaf node every data point is of the same type,
		// which is the type returned by the classifier.
		type = types(0, 0);
	}
	else if (attributesChecked >= data.cols())
	{
		// Case 2: We've checked all the attributes
		// and we still can't build a perfect classifier

		// Return the most likely type
		auto uniqueTypes
			= getUniqueValues(colToStdVector(types.cast<Decimal>()));
		std::vector<size_t> counts(uniqueTypes.size(), 0);
		for (auto i = 0; i < uniqueTypes.size(); ++i)
		{
			for (auto j = 0; j < types.size(); ++j)
			{
				if (types[j] == uniqueTypes[i])
				{
					++counts[i];
				}
			}
		}
		type = uniqueTypes[*std::max_element(cbegin(counts), cend(counts))];
	}
	else
	{
		// Case 3: We can improve by checking more attributes

		// Find the most informative attribute to base the children on
		attributeIndex = bestAttribute(types, data);

		// We'll need a new child for every unique value in the column
		auto uniqueValues = getUniqueValues(
				colToStdVector(data.col(attributeIndex)));

		// Make the children
		for (auto val : uniqueValues)
		{
			// Calculate how big the data subset will be
			auto subsetSize = 0;
			for (auto i = 0; i < data.rows(); ++i)
			{
				if (data(i, attributeIndex) == val)
				{
					++subsetSize;
				}
			}

			assert(subsetSize > 0);

			// Populate the subset with the right rows from the original
			DataMatrix subsetData{subsetSize, data.cols()};
			TypeVector subsetTypes{subsetSize, 1};
			auto insertIndex = 0;
			for (auto i = 0; i < data.rows(); ++i)
			{
				if (data(i, attributeIndex) == val)
				{
					subsetTypes.row(insertIndex) = types.row(i);
					subsetData.row(insertIndex) = data.row(i);
					++insertIndex;
				}
			}

			children.emplace_front(val, this, subsetTypes, subsetData,
					attributesChecked + 1, dt);
		}
	}
}

/**
 * Tells you what class a data point belongs to.
 */
uint8_t DecisionTree::classify(const RowVector& dataPoint) const
{
	const auto *node = root.get();

	std::cout << "Classifying" << std::endl;
	std::cout << dataPoint.cast<int>();
	std::cout << std::endl;

	while (node != nullptr)
	{
		bool foundChild = false;

		// Stop if the current node has no children
		if (node->children.size() == 0)
		{
			assert(node->type != NoType);
			return node->type;
		}

		std::cout << "Node with AttrIndex = " << node->attributeIndex << " looking for child with parentAttrValue = " << static_cast<int>(dataPoint[node->attributeIndex]) << std::endl;
		// Advance to the correct child, based on the data point we have
		for (const auto& child : node->children)
		{
			assert(child.parentAttrValue != NoParentAttrValue);
			assert(node->attributeIndex != NoAttrIndex);

			std::cout << "  Examining child with attributeIndex = " << child.attributeIndex << " and parentVal = " << static_cast<int>(child.parentAttrValue) << std::endl;
			if (child.parentAttrValue == dataPoint[node->attributeIndex])
			{
				node = &child;
				foundChild = true;
				break;
			}
		}

		// If we reach here and there was no correct child... something broke
		assert(foundChild);
	}

	// Failed to find a type
	assert(false);
	return NoType;
}

std::string DecisionTree::Node::name() const
{
	std::stringstream s;
	s << nodeNumber;
	return s.str();
}

std::ostream& DecisionTree::Node::print(std::ostream& out) const
{
	if (children.size() == 0)
	{
		out << nodeNumber << " [label=\"Type "
			<< static_cast<int>(type) << "\"]" << std::endl;
	}
	else
	{
		out << nodeNumber << " [label=\"Attr "
					<< static_cast<int>(attributeIndex) << "\"]"
					<< std::endl;

		for (const auto& child : children)
		{
			out << nodeNumber
			    << " -> " << child.nodeNumber
			    << " [label=\" =" << static_cast<int>(child.parentAttrValue)
				<< "\"]" << std::endl;
			child.print(out);
		}
	}
	return out;
}

std::ostream& DecisionTree::print(std::ostream& out) const
{
	out << "digraph DT {" << std::endl
	    << "    node [fontname=\"Arial\"];" << std::endl;
	root->print(out);
	out << "}" << std::endl;
	return out;
}

std::ostream& operator<<(std::ostream& out, const DecisionTree& dt)
{
	return dt.print(out);
}

/**
 * Calculates the entropy of a dataset by splitting it into two parts
 * based on class and seeing what proportion of the data belongs to each.
 */
double entropy(const TypeVector& types)
{
	assert(types.rows() > 0);

	std::vector<uint8_t> sortedTypes(types.data(),
			types.data() + types.rows());
	std::sort(begin(sortedTypes), end(sortedTypes));

	double ret = 0;
	auto currentType = sortedTypes[0];
	auto currentTypeMatches = 0;
	for (auto type : sortedTypes)
	{
		if (currentType == type)
		{
			++currentTypeMatches;
		}
		else
		{
			// Every time we see a new class, add the entropy from the last one
			auto p = static_cast<double>(currentTypeMatches) / types.rows();
			ret -= (p) * log2(p);
			currentTypeMatches = 1;
			currentType = type;
		}
	}

	// Add the entropy from the final class
	auto p = static_cast<double>(currentTypeMatches) / types.rows();
	ret -= p * log2(p);

	return ret;
}

double entropy(const std::vector<uint8_t>& types)
{
	TypeVector tv(types.size());
	for (auto i = 0; i < types.size(); ++i)
	{
		tv[i] = types[i];
	}
	return entropy(tv);
}

/*
 * Calculates how many bits you will save by knowing the value of an attribute.
 * It's a measure of how well the given column of your data can predict
 * the type.
 */
double gain(const TypeVector& types, const ColVector& dataColumn)
{
	assert(types.rows() > 0);
	assert(types.rows() == dataColumn.rows());

	// Find all the unique values in the column
	auto uniqueValues = getUniqueValues(colToStdVector(dataColumn));

	// Gain is entropy(types) - something per each unique value
	double ret = entropy(types);

	for (auto value : uniqueValues)
	{
		// Select all the data points where that column = that value
		std::vector<uint8_t> subsetTypes;
		for (auto i = 0; i < dataColumn.rows(); ++i)
		{
			if (dataColumn[i] == value)
			{
				subsetTypes.push_back(types[i]);
			}
		}

		assert(subsetTypes.size() > 0);

		// Add the entropy gained by knowing that column = that value
		ret -= static_cast<double>(subsetTypes.size())/dataColumn.rows()
				* entropy(subsetTypes);
	}

	return ret;
}

/*
 * Gives you the 0-based index of the column that maximizes information gain.
 */
size_t bestAttribute(const TypeVector& types, const DataMatrix& data)
{
	double maxGain = -999;
	size_t ret = 999;
	for (auto i = 0; i < data.cols(); ++i)
	{
		auto colGain = gain(types, data.col(i));
		if (colGain > maxGain)
		{
			ret = i;
			maxGain = colGain;
		}
	}
	return ret;
}

std::vector<uint8_t> colToStdVector(const ColVector& col)
{
	return std::vector<uint8_t>(col.data(), col.data() + col.rows());
}

std::vector<uint8_t> rowToStdVector(const RowVector& row)
{
	return std::vector<uint8_t>(row.data(), row.data() + row.cols());
}

/**
 * Utility function to get the unique values from a vector.
 *
 * Unlike std::unique, data does not have to be in sorted order.
 */
std::vector<uint8_t> getUniqueValues(std::vector<uint8_t> vec)
{
	std::sort(begin(vec), end(vec));
	vec.erase(std::unique(begin(vec), end(vec)), end(vec));
	return vec;
}


