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

// Best Attribute tests

TEST(BestAttributeTests, OneColumn)
{
	TypeVector testTypes{1, 1};
	DataMatrix testData{1, 1};
	testTypes << 1;
	testData << 1;

	EXPECT_EQ(0, bestAttribute(testTypes, testData));
}

TEST(BestAttributeTests, TwoColumns)
{
	TypeVector testTypes1{3, 1};
	DataMatrix testData1{3, 2};
	testTypes1 << 1, 2, 1;
	testData1 << 5, 10,
			    6, 10,
				5, 10;

	EXPECT_EQ(0, bestAttribute(testTypes1, testData1));

	TypeVector testTypes2{3, 1};
	DataMatrix testData2{3, 2};
	testTypes2 << 1, 2, 1;
	testData2 << 10, 5,
			    10, 6,
				10, 5;

	EXPECT_EQ(1, bestAttribute(testTypes2, testData2));
}

// Decision Tree tests

TEST(DecisionTreeTests, OneMinimalClass)
{
	TypeVector testTypes{1, 1};
	DataMatrix testData{1, 1};
	testTypes << 1;
	testData << 1;

	DecisionTree dt{testTypes, testData};
	EXPECT_EQ(1, dt.classify(testData.row(0)));
}

TEST(DecisionTreeTests, OneClass)
{
	TypeVector testTypes{5, 1};
	DataMatrix testData{5, 5};
	testTypes << 1, 1, 1, 1, 1;
	testData << 1, 2, 3, 4, 5,
			    1, 3, 3, 9, 10,
				1, 1, 1, 14, 15,
				16, 3, 3, 19, 20,
				21, 4, 5, 24, 25;

	DecisionTree dt{testTypes, testData};
	EXPECT_EQ(1, dt.classify(testData.row(0)));
	EXPECT_EQ(1, dt.classify(testData.row(1)));
	EXPECT_EQ(1, dt.classify(testData.row(2)));
	EXPECT_EQ(1, dt.classify(testData.row(3)));
	EXPECT_EQ(1, dt.classify(testData.row(4)));
}

TEST(DecisionTreeTests, TwoClasses)
{
	TypeVector testTypes{5, 1};
	DataMatrix testData{5, 5};
	testTypes << 1, 1, 1, 2, 2;
	testData << 1, 3, 3, 4, 5,
			    1, 4, 3, 3, 2,
				1, 4, 1, 2, 1,
				4, 2, 1, 1, 3,
				3, 2, 5, 4, 4;

	DecisionTree dt{testTypes, testData};
	EXPECT_EQ(1, dt.classify(testData.row(0)));
	EXPECT_EQ(1, dt.classify(testData.row(1)));
	EXPECT_EQ(1, dt.classify(testData.row(2)));
	EXPECT_EQ(2, dt.classify(testData.row(3)));
	EXPECT_EQ(2, dt.classify(testData.row(4)));
}
