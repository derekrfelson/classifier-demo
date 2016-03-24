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

	testData2(0,0) = 5.0;
	testData2(0,1) = 4.0;
	testData2(0,2) = 3.0;
	testData2(1,0) = 5.0;
	testData2(1,1) = 4.0;
	testData2(1,2) = 3.0;
	testData2(2,0) = 5.0;
	testData2(2,1) = 4.0;
	testData2(2,2) = 3.0;
	testTypes2[0] = 7;
	testTypes2[1] = 7;
	testTypes2[2] = 7;

	EXPECT_EQ(0, entropy(testData1, testTypes1));
	EXPECT_EQ(0, entropy(testData2, testTypes2));
}
