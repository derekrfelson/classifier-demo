#include <gtest/gtest.h>
#include "../src/DecisionTree.h"
#include "../src/Dataset.h"

using DataMatrix = Dataset::DataMatrix;
using TypeVector = Dataset::TypeVector;

TEST(DecisionTreeTests, EntropyZeroWithOnlyOneClass)
{
	TypeVector testTypes1{1, 1};
	testTypes1 << 1;

	TypeVector testTypes2{5, 1};
	testTypes2 << 7, 7, 7, 7, 7;

	EXPECT_EQ(0, entropy(testTypes1));
	EXPECT_EQ(0, entropy(testTypes2));
}

TEST(DecisionTreeTests, ModerateEntropyTwoClasses)
{
	TypeVector testTypes1{14, 1};
	testTypes1 << 1, 1, 1, 1, 1, 1, 1, 1, 1,
			      2, 2, 2, 2, 2;

	EXPECT_NEAR(.94028, entropy(testTypes1), .0001);
}

TEST(DecisionTreeTests, MaxEntropyTwoClasses)
{
	TypeVector testTypes1{2, 1};
	testTypes1 << 1, 2;

	TypeVector testTypes2{4, 1};
	testTypes2 << 1, 2, 2, 1;

	EXPECT_EQ(1, entropy(testTypes1));
	EXPECT_EQ(1, entropy(testTypes2));
}

TEST(DecisionTreeTests, MaxEntropyMultiClass)
{
	TypeVector testTypes1{4, 1};
	testTypes1 << 1, 2, 3, 4;

	EXPECT_EQ(2, entropy(testTypes1));
}

TEST(DecisionTreeTests, MinEntropyMultiClass)
{
	TypeVector testTypes1{4, 1};
	testTypes1 << 1, 1, 1, 1;

	EXPECT_EQ(0, entropy(testTypes1));
}

TEST(DecisionTreeTests, ModerateEntropyMultiClass)
{
	TypeVector testTypes1{14, 1};
	testTypes1 << 1, 1, 1, 1, 1, 1, 1, 1, 1,
			      2, 2, 2, 2, 2;

	EXPECT_NEAR(.94028, entropy(testTypes1), .0001);
}

