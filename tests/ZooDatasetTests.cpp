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
		auto partition = zooData.partition(i);
		EXPECT_EQ(100, partition.training.size());
		EXPECT_EQ(1, partition.testing.size());
	}

	auto partition = zooData.partition(100);
	EXPECT_EQ(100, partition.training.size());
	EXPECT_EQ(1, partition.testing.size());
}
