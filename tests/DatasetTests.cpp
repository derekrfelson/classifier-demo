#include <gtest/gtest.h>
#include "../src/Dataset.h"
#include "../src/Partition.h"
#include "../src/Classifier.h"
#include "../src/CsvReader.h"
#include <iostream>

TEST(DatasetTests, SizeOfDatafile)
{
	auto data = readWineDataset("../data/wine.csv");
	EXPECT_EQ(178, data.size());
}

TEST(DatasetTests, PartitionLeavingOneOut)
{
	auto data = readWineDataset("../data/wine.csv");

	for (auto i = 0; i < 3; ++i)
	{
		auto partition = data.partition(i, i);
		EXPECT_EQ(177, partition.training.size());
		EXPECT_EQ(1, partition.testing.size());
	}

	auto partition = data.partition(177, 177);
	EXPECT_EQ(177, partition.training.size());
	EXPECT_EQ(1, partition.testing.size());
}

TEST(DatasetTests, PartitionForKFold)
{
	auto data = readWineDataset("../data/wine.csv");

	for (auto i = 1; i < 3; ++i)
	{
		auto indices = kFoldIndices(i, 4, data.size());
		auto partition = data.partition(indices.first, indices.second);
		EXPECT_EQ(178-44, partition.training.size());
		EXPECT_EQ(44, partition.testing.size());
	}

	auto indices = kFoldIndices(4, 4, data.size());
	auto partition = data.partition(indices.first, indices.second);
	EXPECT_EQ(178-46, partition.training.size());
	EXPECT_EQ(46, partition.testing.size());
}
