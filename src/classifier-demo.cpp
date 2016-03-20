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
#include "Classifier.h"
#include "Partition.h"
#include "ZooDataset.h"

using CovarianceMatrix = ZooDataset::CovarianceMatrix;
using MeanRowVector = ZooDataset::MeanRowVector;

int main(int argc, char** argv)
{
	ZooDataset zooData{"../data/zoo.csv"};
	std::vector<Classifier> classes;
	classes.reserve(ZooDataset::NumClasses);
	std::vector<ZooDataset> testingData;
	testingData.reserve(ZooDataset::NumClasses);

	for (auto i = 1; i <= ZooDataset::NumClasses; ++i)
	{
		auto classSubset = zooData.getSubsetByClass(i);
		std::cout << "size: " << classSubset.size() << std::endl;
		auto indices = kFoldIndices(1, 10, classSubset.size());
		auto partitions = zooData.partition(indices.first, indices.second);
		Classifier c{partitions.training};
		classes.push_back(c);
		testingData.push_back(partitions.testing);
	}

	return 0;
}

