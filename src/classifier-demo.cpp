#include <iostream>
#include "Classifier.h"
#include "Partition.h"
#include "Dataset.h"
#include "ZooDataset.h"

using CovarianceMatrix = Dataset::CovarianceMatrix;
using MeanRowVector = Dataset::RowVector;

void classifyAndTest(const Dataset& data,
		unsigned int numFolds,
		ClassifierType ctype);

int main(int argc, char** argv)
{
	auto zooData = readZooDataset("../data/zoo.csv");

	std::cout << "Zoo data using 10-fold cross-validation "
			  << "(Optimal Bayes classifier)" << std::endl << std::endl;
	classifyAndTest(zooData, 10, ClassifierType::OPTIMAL);

	std::cout << std::endl;
	std::cout << "Zoo data using leave-one-out cross-validation "
			  << "(Optimal Bayes classifier)" << std::endl << std::endl;
	classifyAndTest(zooData, zooData.size(), ClassifierType::OPTIMAL);

	std::cout << std::endl;
	std::cout << "Zoo data using 10-fold cross-validation "
			  << "(Naive Bayes classifier)" << std::endl << std::endl;
	classifyAndTest(zooData, 10, ClassifierType::NAIVE);

	std::cout << std::endl;
	std::cout << "Zoo data using leave-one-out cross-validation "
	 	      << "(Naive Bayes classifier)" << std::endl << std::endl;
	classifyAndTest(zooData, zooData.size(), ClassifierType::NAIVE);

	std::cout << std::endl;
	std::cout << "Zoo data using 10-fold cross-validation "
			  << "(Linear Bayes classifier)" << std::endl << std::endl;
	classifyAndTest(zooData, 10, ClassifierType::LINEAR);

	std::cout << std::endl;
	std::cout << "Zoo data using leave-one-out cross-validation "
	 	      << "(Linear Bayes classifier)" << std::endl << std::endl;
	classifyAndTest(zooData, zooData.size(), ClassifierType::LINEAR);

	return 0;
}

void classifyAndTest(const Dataset& data,
		unsigned int numFolds,
		ClassifierType ctype)
{
	std::vector<unsigned int> timesRight(numFolds, 0);
	std::vector<unsigned int> timesWrong(numFolds, 0);
	auto totalTimesRight = 0;
	auto totalTimesWrong = 0;

	// Classify and test the data
	for (auto k = 1; k <= numFolds; ++k)
	{
		// Partition into testing and training sets
		auto indices = kFoldIndices(k, numFolds, data.size());
		auto partitions = data.partition(indices.first, indices.second);

		// Create a classifier for the dataset
		auto c = partitions.training.classifier(ctype);

		// Test each point in the testing set
		for (auto i = 0; i < partitions.testing.size(); ++i)
		{
			// Classify
			auto type = c.classify(partitions.testing.getPoint(i));

			std::cout << "Decided class " << static_cast<int>(type)
					<< " for " << partitions.testing.getName(i) << " (actual "
					<< static_cast<int>(partitions.testing.getType(i))
					<< "): " << partitions.testing.getPoint(i) << std::endl;

			// Update counters
			if (type == partitions.testing.getType(i))
			{
				++timesRight[k-1];
			}
			else
			{
				++timesWrong[k-1];
			}
		}

		// Report the accuracy on this fold
		std::cout << "Fold " << k << ": timesRight=" << timesRight[k-1]
		          << ", timesWrong=" << timesWrong[k-1]
		          << ", accuracy: " << timesRight[k-1]/
					 static_cast<double>(timesWrong[k-1]+timesRight[k-1])
				  << std::endl << std::endl;

		// Update overall accuracy
		totalTimesRight += timesRight[k-1];
		totalTimesWrong += timesWrong[k-1];
	}

	// Report overall accuracy
	std::cout << "Total: timesRight=" << totalTimesRight
			  << ", timesWrong=" << totalTimesWrong
			  << ", accuracy=" << totalTimesRight
			      / static_cast<double>(totalTimesRight + totalTimesWrong)
			  << std::endl;
}

