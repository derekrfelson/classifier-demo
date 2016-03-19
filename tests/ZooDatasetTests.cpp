#include <gtest/gtest.h>
#include "../src/ZooDataset.h"
#include "../src/Partition.h"

TEST(ZooDatasetTests, SizeOfDatafile)
{
	auto zooData = ZooDataset{"../data/zoo.csv"};
	EXPECT_EQ(101, zooData.size());
}

TEST(ZooDatasetTests, PartitionLeavingOneOut)
{
	auto zooData = ZooDataset{"../data/zoo.csv"};

	for (auto i = 0; i < 3; ++i)
	{
		auto partition = zooData.partition(i, i);
		EXPECT_EQ(100, partition.training.size());
		EXPECT_EQ(1, partition.testing.size());
	}

	auto partition = zooData.partition(100, 100);
	EXPECT_EQ(100, partition.training.size());
	EXPECT_EQ(1, partition.testing.size());
}

TEST(ZooDatasetTests, PartitionForKFold)
{
	auto zooData = ZooDataset{"../data/zoo.csv"};

	for (auto i = 1; i < 5; ++i)
	{
		auto indices = kFoldIndices(i, 5, zooData.size());
		auto partition = zooData.partition(indices.first, indices.second);
		EXPECT_EQ(81, partition.training.size());
		EXPECT_EQ(20, partition.testing.size());
	}

	auto indices = kFoldIndices(5, 5, zooData.size());
	auto partition = zooData.partition(indices.first, indices.second);
	EXPECT_EQ(80, partition.training.size());
	EXPECT_EQ(21, partition.testing.size());
}
