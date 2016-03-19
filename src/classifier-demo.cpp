#include "Animal.h"
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <array>
#include <memory>
#include "ZooDataset.h"

int main(int argc, char** argv)
{
	auto zooData = ZooDataset{"../data/zoo.csv"};

	//for (auto i = 1; i <= ZooDataset::NumClasses; ++i)
	for (auto i = 1; i <= 1; ++i)
	{
		auto subset = zooData.getSubsetByClass(i);
		auto cmInv = subset.getCovarianceMatrixInverse();
		auto cmDet = subset.getCovarianceMatrixDeterminant();

		std::cout << subset.getMeans() << std::endl << std::endl;
		std::cout << cmInv << std::endl << std::endl;
		std::cout << cmDet << std::endl;
	}

	return 0;
}

