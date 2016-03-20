#include "Animal.h"
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <array>
#include <memory>
#include <limits>
#include "Classifier.h"
#include "Partition.h"
#include "ZooDataset.h"

using CovarianceMatrix = ZooDataset::CovarianceMatrix;
using MeanRowVector = ZooDataset::RowVector;

int main(int argc, char** argv)
{
	ZooDataset zooData{"../data/zoo.csv"};

	constexpr auto numFolds = 10;

	for (auto k = 1; k <= numFolds; ++k)
	{
		// Partition into testing and training sets
		auto indices = kFoldIndices(k, numFolds, zooData.size());
		auto partitions = zooData.partition(indices.first, indices.second);

		// Create a classifier for the dataset
		Classifier c{partitions.training};

		// Test each point in the testing set
		for (auto i = 0; i < partitions.testing.size(); ++i)
		{
			auto type = c.classify(partitions.testing.getPoint(i));
			std::cout << "Decided class " << static_cast<int>(type)
					<< " for " << partitions.testing.getName(i) << " (actual "
					<< static_cast<int>(partitions.testing.getType(i))
					<< "): " << partitions.testing.getPoint(i) << std::endl;
		}
	}

	return 0;
}

