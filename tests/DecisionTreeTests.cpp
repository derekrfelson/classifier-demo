#include <gtest/gtest.h>
#include "../src/DecisionTree.h"
#include "../src/Dataset.h"

using DataMatrix = Dataset::DataMatrix;
using TypeVector = Dataset::TypeVector;

TEST(DecisionTreeTests, EntropyZeroWithOnlyOneClass)
{
	DataMatrix testData1{1, 1};
	TypeVector testTypes1{1, 1};

	testData1(0,0) = 5.0;
	testTypes1[0] = 1;

	DataMatrix testData2{5, 3};
	TypeVector testTypes2{5, 1};

	testData2 << 5, 4, 3,
			     5, 4, 3,
				 5, 4, 3,
				 5, 4, 3,
				 5, 4, 3;

	testTypes2 << 7, 7, 7, 7, 7;

	EXPECT_EQ(0, entropy(testData1, testTypes1));
	EXPECT_EQ(0, entropy(testData2, testTypes2));
}

TEST(DecisionTreeTests, MaxEntropy)
{
	DataMatrix testData1{2, 1};
	TypeVector testTypes1{2, 1};

	testData1 << 5, 4;
	testTypes1 << 1, 2;

	DataMatrix testData2{5, 3};
	TypeVector testTypes2{5, 1};

	testData2 << 5, 4, 3,
			     5, 4, 3,
				 5, 4, 3,
				 5, 4, 3,
				 5, 4, 3;

	testTypes2 << 1, 2, 3, 2, 1;

	EXPECT_EQ(1, entropy(testData1, testTypes1));
	EXPECT_EQ(1, entropy(testData2, testTypes2));
}
