#include <iostream>
#include <array>
#include "Classifier.h"
#include "Partition.h"
#include "ZooDataset.h"

using CovarianceMatrix = ZooDataset::CovarianceMatrix;
using MeanRowVector = ZooDataset::RowVector;

int main(int argc, char** argv)
{
	ZooDataset zooData{"../data/zoo.csv"};

	constexpr auto numFolds = 10;

	std::array<unsigned int, numFolds> timesRight;
	std::array<unsigned int, numFolds> timesWrong;

	for (auto k = 1; k <= numFolds; ++k)
	{
		timesRight[k-1] = 0;
		timesWrong[k-1] = 0;
		std::cout << "Fold " << k << std::endl;

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

			if (type == partitions.testing.getType(i))
			{
				++timesRight[k-1];
			}
			else
			{
				++timesWrong[k-1];
			}
		}
	}

	auto totalTimesRight = 0;
	auto totalTimesWrong = 0;

	for (auto k = 1; k <= numFolds; ++k)
	{
		std::cout << "Fold " << k << ": timesRight=" << timesRight[k-1]
                  << ", timesWrong=" << timesWrong[k-1]
                  << ", accuracy: " << timesRight[k-1]/
				      static_cast<double>(timesWrong[k-1]+timesRight[k-1])
			      << std::endl;
		totalTimesRight += timesRight[k-1];
		totalTimesWrong += timesWrong[k-1];
	}

	std::cout << "Total: timesRight=" << totalTimesRight
			  << ", timesWrong=" << totalTimesWrong
			  << ", accuracy=" << totalTimesRight
			      / static_cast<double>(totalTimesRight + totalTimesWrong)
			  << std::endl;

	return 0;
}

