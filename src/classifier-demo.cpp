#include <iostream>
#include <string>
#include <array>
#include "BayesClassifier.h"
#include "Partition.h"
#include "Dataset.h"
#include "Types.h"
#include "CsvReader.h"

void classifyAndTest(const Dataset& data,
		unsigned int numFolds,
		ClassifierType ctype,
		int verbosity);

int main(int argc, char** argv)
{
	auto verbosity = argc - 1;

	std::array<Dataset, 3> datasets {
			readIrisDataset("../data/iris.csv"),
			readHeartDiseaseDataset("../data/heartDisease.csv"),
			readWineDataset("../data/wine.csv")
	};

	datasets[0].shuffle();
	datasets[1].shuffle();
	datasets[2].shuffle();

	std::array<std::string, 3> datasetLabels = {
			"Iris", "Heart Disease", "Wine"
	};

	std::array<ClassifierType, 3> classifierTypes = {
			ClassifierType::OPTIMAL,
			ClassifierType::NAIVE,
			ClassifierType::LINEAR
	};

	std::array<std::string, 3> classifierTypeLabels = {
			"Optimal",
			"Naive",
			"Linear"
	};

	for (auto datasetNum = 0; datasetNum < 3; ++datasetNum)
	{
		for (auto classifierNum = 0; classifierNum < 3; ++classifierNum)
		{
			std::cout << datasetLabels[datasetNum]
					  << " data using 10-fold cross-validation "
					  << "(" << classifierTypeLabels[classifierNum]
					  << " Bayes classifier)"
					  << std::endl << std::endl;
			classifyAndTest(datasets[datasetNum],
					10,
					classifierTypes[classifierNum],
					verbosity);

			std::cout << datasetLabels[datasetNum]
					  << " data using leave-one-out cross-validation "
					  << "(" << classifierTypeLabels[classifierNum]
					  << " Bayes classifier)"
					  << std::endl << std::endl;
			classifyAndTest(datasets[datasetNum],
					datasets[datasetNum].size(),
					classifierTypes[classifierNum],
					verbosity);
		}
	}

	return 0;
}

void classifyAndTest(const Dataset& data,
		unsigned int numFolds,
		ClassifierType ctype,
		int verbosity)
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

			if (verbosity > 1)
			{
				std::cout << "Decided class " << static_cast<int>(type)
					<< " for " << partitions.testing.getName(i) << " (actual "
					<< static_cast<int>(partitions.testing.getType(i))
					<< "): " << partitions.testing.getPoint(i) << std::endl;
			}

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

		// Report the accuracy on this fold (unless we're just doing 1 element)
		if (partitions.testing.size() > 1 && verbosity > 0)
		{
			std::cout << "Fold " << k << ": timesRight=" << timesRight[k-1]
					  << ", timesWrong=" << timesWrong[k-1]
		              << ", accuracy: " << timesRight[k-1]/
					     static_cast<double>(timesWrong[k-1]+timesRight[k-1])
				      << std::endl << std::endl;
		}

		// Update overall accuracy
		totalTimesRight += timesRight[k-1];
		totalTimesWrong += timesWrong[k-1];
	}

	// Report overall accuracy
	std::cout << "Total: timesRight=" << totalTimesRight
			  << ", timesWrong=" << totalTimesWrong
			  << ", accuracy=" << totalTimesRight
			      / static_cast<double>(totalTimesRight + totalTimesWrong)
			  << std::endl << std::endl << std::endl;
}

