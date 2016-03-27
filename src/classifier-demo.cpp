#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include "BayesClassifier.h"
#include "DecisionTree.h"
#include "Partition.h"
#include "Dataset.h"
#include "Types.h"
#include "CsvReader.h"

void classifyAndTest(const Dataset& data,
		unsigned int numFolds,
		ClassifierType ctype,
		int verbosity,
		std::ostream& resultsOut,
		std::string modelOutName,
		std::ostream& finalResults);

int main(int argc, char** argv)
{
	auto verbosity = argc - 1;

	std::array<Dataset, 3> datasets {
			readIrisDataset("../data/iris.csv"),
			readHeartDiseaseDataset("../data/heartDisease.csv"),
			readWineDataset("../data/wine.csv")
	};

	std::array<Dataset, 3> discreteDatasets {
		readIrisDataset("../data/irisDiscrete.csv"),
		readHeartDiseaseDataset("../data/heartDiseaseDiscrete.csv"),
		readWineDataset("../data/wineDiscrete.csv")
	};

	datasets[0].shuffle();
	datasets[1].shuffle();
	datasets[2].shuffle();
	discreteDatasets[0].shuffle();
	discreteDatasets[1].shuffle();
	discreteDatasets[2].shuffle();

	std::array<std::string, 3> datasetLabels = {
			"Iris", "Heart Disease", "Wine"
	};

	std::array<ClassifierType, 4> classifierTypes = {
			ClassifierType::OPTIMAL,
			ClassifierType::NAIVE,
			ClassifierType::LINEAR,
			ClassifierType::DECISION_TREE
	};

	std::array<std::string, 4> classifierTypeLabels = {
			"Optimal Bayes",
			"Naive Bayes",
			"Linear Bayes",
			"Decision Tree"
	};

	// Open final results CSV
	auto finalResults = std::ofstream{"output/finalResults.txt"};
	assert(finalResults.is_open());

	// Output column labels on final results CSV
	finalResults << "*,";
	for (auto datasetNum = 0; datasetNum < 3; ++datasetNum)
	{
		finalResults << datasetLabels[datasetNum] << " 10-fold,";
		finalResults << datasetLabels[datasetNum] << " leave-one-out,";
	}
	finalResults << std::endl;

	for (auto classifierNum = 0; classifierNum < 4; ++classifierNum)
	{
		// Output row label for final results CSV
		finalResults << classifierTypeLabels[classifierNum];

		for (auto datasetNum = 0; datasetNum < 3; ++datasetNum)
		{
			std::stringstream out2name;
			out2name << "output/"
					 << datasetLabels[datasetNum]
					 << "-" << classifierTypeLabels[classifierNum];

			// 10-fold cross validation
			auto resultsFile = std::ofstream{out2name.str()
				+ "-10fold-results.txt"};
			assert(resultsFile.is_open());
			auto modelFileName  = out2name.str() + "-10fold-model";
			auto& data = classifierNum < 3
					? datasets[datasetNum] : discreteDatasets[datasetNum];
			std::cout << datasetLabels[datasetNum]
					  << " data using 10-fold cross-validation "
					  << "(" << classifierTypeLabels[classifierNum]
					  << " classifier)"
					  << std::endl << std::endl;

			classifyAndTest(data, 10, classifierTypes[classifierNum],
					verbosity, resultsFile, modelFileName, finalResults);
			resultsFile.close();

			// Leave-one-out cross validation
			resultsFile = std::ofstream{out2name.str()
				+ "-leaveOneOut-results.txt"};
			assert(resultsFile.is_open());
			modelFileName  = out2name.str() + "-leaveOneOut-model";
			std::cout << datasetLabels[datasetNum]
					  << " data using leave-one-out cross-validation "
					  << "(" << classifierTypeLabels[classifierNum]
					  << " classifier)"
					  << std::endl << std::endl;
			classifyAndTest(data, datasets[datasetNum].size(),
					classifierTypes[classifierNum], verbosity,
					resultsFile, modelFileName, finalResults);
			resultsFile.close();
		}

		finalResults << std::endl;
	}

	finalResults.close();

	return 0;
}

void classifyAndTest(const Dataset& data,
		unsigned int numFolds,
		ClassifierType ctype,
		int verbosity,
		std::ostream& resultsOut,
		std::string modelOutName,
		std::ostream& finalResults)
{
	std::vector<unsigned int> timesRight(numFolds, 0);
	std::vector<unsigned int> timesWrong(numFolds, 0);
	std::vector<unsigned int> timesUndecided(numFolds, 0);
	auto totalTimesRight = 0;
	auto totalTimesWrong = 0;
	auto totalTimesUndecided = 0;

	// Classify and test the data
	for (auto k = 1; k <= numFolds; ++k)
	{
		// Partition into testing and training sets
		auto indices = kFoldIndices(k, numFolds, data.size());
		auto partitions = data.partition(indices.first, indices.second);

		// Create a classifier for the dataset
		auto c = partitions.training.classifier(ctype);

		// If it's a decision tree, output it
		if (ctype == ClassifierType::DECISION_TREE)
		{
			std::stringstream name;
			name << modelOutName << "-" << k << ".dot";
			auto modelOut = std::ofstream{name.str()};
			assert(modelOut.is_open());
			dynamic_cast<DecisionTree*>(c.get())->print(modelOut);
			modelOut.close();
		}

		// Test each point in the testing set
		for (auto i = 0; i < partitions.testing.size(); ++i)
		{
			// Classify
			auto type = c->classify(partitions.testing.getPoint(i));

			resultsOut << "Decided class " << static_cast<int>(type)
					<< " for " << partitions.testing.getName(i) << " (actual "
					<< static_cast<int>(partitions.testing.getType(i))
					<< "): " << partitions.testing.getPoint(i) << std::endl;

			// Update counters
			if (type == partitions.testing.getType(i))
			{
				++timesRight[k-1];
			}
			else if (type == NoType)
			{
				++timesUndecided[k-1];
			}
			else
			{
				++timesWrong[k-1];
			}
		}

		// Report the accuracy on this fold (unless we're just doing 1 element)
		if (partitions.testing.size() > 1)
		{
			resultsOut << "Fold " << k << ": timesRight=" << timesRight[k-1]
					  << ", timesWrong=" << timesWrong[k-1]
					  << ", timesUndecided=" << timesUndecided[k-1]
		              << ", accuracy: " << timesRight[k-1]/
					     static_cast<double>(timesWrong[k-1]
										+timesRight[k-1]+timesUndecided[k-1])
				      << std::endl << std::endl;
		}

		// Update overall accuracy
		totalTimesRight += timesRight[k-1];
		totalTimesUndecided += timesUndecided[k-1];
		totalTimesWrong += timesWrong[k-1];
	}

	// Report overall accuracy
	auto accuracy = totalTimesRight / static_cast<double>(totalTimesRight
		    		  + totalTimesWrong + totalTimesUndecided);
	resultsOut << "Total: timesRight=" << totalTimesRight
			  << ", timesWrong=" << totalTimesWrong
			  << ", timesUndecided=" << totalTimesUndecided
			  << ", accuracy=" << accuracy
			  << std::endl << std::endl << std::endl;

	finalResults << "," << accuracy;
}

