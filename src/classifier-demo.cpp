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

template <typename T>
std::pair<std::vector<T>, std::vector<T> >
partition(const std::vector<T>& dataset, size_t start, size_t length)
{
	auto ret = std::pair<std::vector<T>, std::vector<T> >{};
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

std::array<std::vector<Animal>, 7>
partitionByClass(const std::vector<Animal>& dataset)
{
	auto ret = std::array<std::vector<Animal>, 7>{};
	for (auto i = 0; i < 7; ++i)
	{
		for (const auto& animal : dataset)
		{
			if (animal.type == i + 1)
			{
				ret[i].push_back(animal);
			}
		}
	}
	return ret;
}

int main(int argc, char** argv)
{
	std::ifstream file("../data/zoo.csv");
	std::string line;
	std::vector<Animal> animals;

	while(std::getline(file, line))
	{
		animals.emplace_back(Animal{line});
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
