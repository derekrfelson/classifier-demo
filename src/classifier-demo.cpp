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

template <typename T>
using PData = std::shared_ptr<T>;

template <typename T>
using Dataset = std::vector<PData<T> >;

template <typename T>
using Partition = std::pair<Dataset<T>, Dataset<T> >;

template <typename T>
Partition<T> partition(const Dataset<T>& dataset, size_t start, size_t length)
{
	auto ret = Partition<T>{};
	auto i = 0;
	for (; i < start; ++i)
	{
		ret.first.push_back(dataset[i]);
	}
	for (; i < start + length; ++i)
	{
		ret.second.push_back(dataset[i]);
	}
	for (; i < dataset.size(); ++i)
	{
		ret.first.push_back(dataset[i]);
	}
	assert(ret.first.size() + ret.second.size() == dataset.size());
	return ret;
}

std::array<Dataset<Animal>, 7>
partitionByClass(const Dataset<Animal>& dataset)
{
	auto ret = std::array<Dataset<Animal>, 7>{};
	for (auto i = 0; i < 7; ++i)
	{
		for (const auto& animal : dataset)
		{
			if (animal->type == i + 1)
			{
				ret[i].push_back(animal);
			}
		}

		// Terminate the program if we have an empty class
		assert(ret[i].size() > 0);
	}

	return ret;
}

int main(int argc, char** argv)
{
	std::ifstream file("../data/zoo.csv");
	std::string line;
	std::vector<std::shared_ptr<Animal> > animals;

	while(std::getline(file, line))
	{
		animals.emplace_back(std::make_shared<Animal>(line));
	}

	auto p = partition<Animal>(animals, 0, std::ceil(animals.size() / 10.0));

	std::cout << "Training set size: " << p.first.size() << std::endl;
	std::cout << "Testing set size: " << p.second.size() << std::endl;

	auto classes = partitionByClass(p.first);
	for (const auto& c : classes)
	{
		std::cout << c.size() << std::endl;
	}

	return 0;
}
