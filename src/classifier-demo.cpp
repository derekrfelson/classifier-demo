#include "Animal.h"
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>

int main(int argc, char** argv)
{
	std::ifstream file("../data/zoo.csv");
	std::string line;
	std::string field;

	while(std::getline(file, line))
	{
		std::cout << line << std::endl;
		Animal a{line};
	}

	return 0;
}


