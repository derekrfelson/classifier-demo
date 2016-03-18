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

	for (auto i = 1; i <= ZooDataset::NumClasses; ++i)
	{
		std::cout << zooData.getSubsetByClass(i).getMeans() << std::endl;
	}

	return 0;
}
