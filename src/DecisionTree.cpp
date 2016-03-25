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
#include <sstream>

constexpr uint8_t NoType = 255;

static std::vector<uint8_t> getUniqueValues(const ColVector& col);

DecisionTree::DecisionTree(const TypeVector& types, const DataMatrix& data)
{
	auto best = bestAttribute(types, data);
	root = std::make_unique<Node>(best, types, data);
}

/**
 * Initialize a root node on the decision tree.
 */
DecisionTree::Node::Node(size_t attributeIndex, const TypeVector& types,
		const DataMatrix& data)
: Node(attributeIndex, NoType, nullptr, types, data)
{
}

/**
 * Initialize a child node on the decision tree.
 *
 * Each node on the tree looks at an attribute, the value of which can
 * fall into several different classes. Specifying the value of the attribute,
 * is the same as choosing one of its child nodes.
 *
 * parentAttrValue: Value matched for the parent's distinguishing attribute
 */
DecisionTree::Node::Node(size_t attributeIndex, uint8_t parentAttrValue,
		const Node* parent, const TypeVector& types, const DataMatrix& data)
: attributeIndex{attributeIndex},
  parentAttrValue{parentAttrValue},
  children{},
  parent{parent},
  type{NoType}
{
	std::unique_ptr<DataMatrix> subsetData;
	std::unique_ptr<TypeVector> subsetTypes;

	// Take the appropriate subsets of the data if not the root
	if (parent != nullptr)
	{
		assert(parentAttrValue != NoType);

		// Calculate how big the data subset will be
		auto subsetSize = 0;
		for (auto i = 0; i < data.rows(); ++i)
		{
			if (data(i, parent->attributeIndex) == parentAttrValue)
			{
				++subsetSize;
			}
		}

		assert(subsetSize > 0);

		// Populate the subset with the right rows from the original
		subsetData = std::make_unique<DataMatrix>(subsetSize, data.cols());
		subsetTypes = std::make_unique<TypeVector>(subsetSize, 1);
		auto insertIndex = 0;
		for (auto i = 0; i < data.rows(); ++i)
		{
			if (data(i, parent->attributeIndex) == parentAttrValue)
			{
				subsetTypes->row(insertIndex) = types.row(i);
				subsetData->row(insertIndex) = data.row(i);
				++insertIndex;
			}
		}
	}
	else
	{
		subsetData = std::make_unique<DataMatrix>(data);
		subsetTypes = std::make_unique<TypeVector>(types);
	}

	// Construct child decision nodes unless we already have only one class
	if (entropy(*subsetTypes) != 0)
	{
		// Find the most informative attribute to base the children on
		auto childAttribute = bestAttribute(*subsetTypes, *subsetData);

		// We'll need a new child for every unique value in the column
		auto uniqueValues = getUniqueValues(subsetData->col(attributeIndex));

		// Make the children
		for (auto val : uniqueValues)
		{
			// You get to the child node by selecting $val for $attribute.
			// The child will then further discriminate based on
			// $childAttribute.
			children.emplace_front(childAttribute, val, this,
					*subsetTypes, *subsetData);
		}
	}
	else
	{
		// In a leaf node every data point is of the same type,
		// which is the type returned by the classifier.
		type = subsetTypes->operator()(0, 0);
	}
}

/**
 * Tells you what class a data point belongs to.
 */
uint8_t DecisionTree::classify(const RowVector& dataPoint) const
{
	const auto *node = root.get();
	while (node != nullptr)
	{
		// Stop if the current node has no children
		if (node->children.size() == 0)
		{
			assert(node->type != NoType);
			return node->type;
		}

		// Advance to the correct child, based on the data point we have
		for (const auto& child : node->children)
		{
			if (child.parentAttrValue == dataPoint[node->attributeIndex])
			{
				node = &child;
			}
		}
	}

	// Failed to find a type
	assert(false);
	return NoType;
}

std::string DecisionTree::Node::name() const
{
	if (children.size() == 0)
	{
		std::stringstream s;
		s << "\"Type " << static_cast<int>(type) << "\"";
		return s.str();
	}
	else
	{
		std::stringstream s;
		s << "\"Attr " << attributeIndex << "\"";
		return s.str();
	}
}

std::ostream& DecisionTree::Node::print(std::ostream& out) const
{
	if (children.size() == 0)
	{
		out << name() << std::endl;
	}
	else
	{
		for (const auto& child : children)
		{
			out << name() << " -> " << child.name()
			    << " [label=\" =" << static_cast<int>(child.parentAttrValue)
				<< "\"" << std::endl;
		}
	}
}

std::ostream& DecisionTree::print(std::ostream& out) const
{
	out << "Digraph BST {" << std::endl
	    << "    node [fontname=\"Arial\"];" << std::endl;
	root->print(out);
	out << "}" << std::endl;
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
	auto uniqueValues = getUniqueValues(dataColumn);

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

/**
 * Utility function to get the unique values from a column vector.
 *
 * Unlike std::unique, data does not have to be in sorted order.
 */
std::vector<uint8_t> getUniqueValues(const ColVector& col)
{
	std::vector<uint8_t> uniqueValues(col.data(), col.data() + col.rows());
	std::sort(begin(uniqueValues), end(uniqueValues));
	uniqueValues.erase(std::unique(begin(uniqueValues), end(uniqueValues)),
			end(uniqueValues));
	return uniqueValues;
}


