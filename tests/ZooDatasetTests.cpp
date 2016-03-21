#include <gtest/gtest.h>
#include "../src/ZooDataset.h"
#include "../src/Partition.h"
#include "../src/Classifier.h"
#include <iostream>

using Decimal = ZooDataset::Decimal;

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

TEST(ZooDatasetTests, Means)
{
	auto zooData = ZooDataset{"../data/zooMeanTest.csv"};
	EXPECT_EQ(4, zooData.size());

	auto means = zooData.getMeans();
	std::array<Decimal, 16> expectedMeans = { 0.75, 0, 0.25, 0.75, 0, 0.25,
			0.75, 1, 1, 0.75, 0, 0.25, 3, 0.5, 0, 0.75};

	for (auto i = 0; i < 16; ++i)
	{
		EXPECT_EQ(expectedMeans[i], means[i]);
	}
}

/*
TEST(ZooDatasetTests, Pseudodeterminant)
{
}

TEST(ZooDatasetTests, CovarianceMatrix)
{
	//auto zooData = ZooDataset{"../data/zoo.csv"};
	//EXPECT_EQ(101, zooData.size());

	//std::cout << zooData.getCovarianceMatrix();
	//std::cout << std::endl;
}

TEST(ZooDatasetTests, InverseCovarianceMatrix)
{
	//auto zooData = ZooDataset{"../data/zoo.csv"};
	//EXPECT_EQ(101, zooData.size());

	//std::cout << zooData.getCovarianceMatrixInverse();
	//std::cout << std::endl;
}
*/
