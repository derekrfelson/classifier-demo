#include <gtest/gtest.h>
#include "../src/DecisionTree.h"
#include "../src/Dataset.h"

using DataMatrix = Dataset::DataMatrix;
using TypeVector = Dataset::TypeVector;

// Entropy tests

TEST(EntropyTests, EntropyZeroWithOnlyOneClass)
{
	TypeVector testTypes1{1, 1};
	testTypes1 << 1;

	TypeVector testTypes2{5, 1};
	testTypes2 << 7, 7, 7, 7, 7;

	EXPECT_EQ(0, entropy(testTypes1));
	EXPECT_EQ(0, entropy(testTypes2));
}

TEST(EntropyTests, ModerateEntropyTwoClasses)
{
	TypeVector testTypes1{14, 1};
	testTypes1 << 1, 1, 1, 1, 1, 1, 1, 1, 1,
			      2, 2, 2, 2, 2;

	EXPECT_NEAR(.94028, entropy(testTypes1), .0001);
}

TEST(EntropyTests, MaxEntropyTwoClasses)
{
	TypeVector testTypes1{2, 1};
	testTypes1 << 1, 2;

	TypeVector testTypes2{4, 1};
	testTypes2 << 1, 2, 2, 1;

	EXPECT_EQ(1, entropy(testTypes1));
	EXPECT_EQ(1, entropy(testTypes2));
}

TEST(EntropyTests, MaxEntropyMultiClass)
{
	TypeVector testTypes1{4, 1};
	testTypes1 << 1, 2, 3, 4;

	EXPECT_EQ(2, entropy(testTypes1));
}

TEST(EntropyTests, MinEntropyMultiClass)
{
	TypeVector testTypes1{4, 1};
	testTypes1 << 1, 1, 1, 1;

	EXPECT_EQ(0, entropy(testTypes1));
}

TEST(EntropyTests, ModerateEntropyMultiClass)
{
	TypeVector testTypes1{14, 1};
	testTypes1 << 1, 1, 1, 1, 1, 1, 1, 1, 1,
			      2, 2, 2, 2, 2;

	EXPECT_NEAR(.94028, entropy(testTypes1), .0001);
}

// Information gain tests

TEST(GainTests, PerfectGain)
{
	TypeVector testTypes{4, 1};
	DataMatrix testData{4, 2};
	testTypes << 1, 1, 2, 2;
	testData << 10, 1,
			    10, 1,
				0, 1,
				0, 1;

	EXPECT_EQ(1, gain(testTypes, testData.col(0)));
}

TEST(GainTests, NoGain)
{
	TypeVector testTypes{4, 1};
	DataMatrix testData{4, 2};
	testTypes << 1, 1, 2, 2;
	testData << 10, 1,
			    10, 1,
				0, 1,
				0, 1;

	EXPECT_EQ(0, gain(testTypes, testData.col(1)));
}

TEST(GainTests, SomeGain)
{
	TypeVector testTypes{14, 1};
	DataMatrix testData{14, 1};
	testTypes << 1, 1, 2, 2, 2,
			     1, 2, 1, 2, 2,
				 2, 2, 2, 1;
	testData << 0, 1, 0, 0, 0,
			    1, 1, 0, 0, 0,
				1, 1, 0, 1;

	EXPECT_NEAR(.048, gain(testTypes, testData.col(0)), .001);
}
