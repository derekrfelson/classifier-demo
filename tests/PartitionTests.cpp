#include <gtest/gtest.h>
#include "../src/Partition.h"

TEST(PartitionTests, KFoldIndicesEvenSplit)
{
	auto res = kFoldIndices(1, 2, 100);
	EXPECT_EQ(0, res.first);
	EXPECT_EQ(49, res.second);
	res = kFoldIndices(2, 2, 100);
	EXPECT_EQ(50, res.first);
	EXPECT_EQ(99, res.second);
}

TEST(PartitionTests, KFoldIndicesUnevenSplit)
{
	auto res = kFoldIndices(1, 3, 100);
	EXPECT_EQ(0, res.first);
	EXPECT_EQ(32, res.second);
	res = kFoldIndices(2, 3, 100);
	EXPECT_EQ(33, res.first);
	EXPECT_EQ(65, res.second);
	res = kFoldIndices(3, 3, 100);
	EXPECT_EQ(66, res.first);
	EXPECT_EQ(99, res.second);
}

TEST(PartitionTests, KFoldIndicesSmallNumbers)
{
	auto res = kFoldIndices(1, 2, 2);
	EXPECT_EQ(0, res.first);
	EXPECT_EQ(0, res.second);
	res = kFoldIndices(2, 2, 2);
	EXPECT_EQ(1, res.first);
	EXPECT_EQ(1, res.second);
}
