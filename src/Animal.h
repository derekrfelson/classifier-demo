/*
 * Animal.h
 *
 *  Created on: Mar 17, 2016
 *      Author: derek
 */

#ifndef ANIMAL_H_
#define ANIMAL_H_

#include <string>
#include <array>
#include <cstdint>

class Animal
{
public:
	explicit Animal(std::string csvLine);
	uint8_t getType() const;
private:
	std::string name;
	std::array<uint8_t, 17> fields;
};

#endif /* ANIMAL_H_ */
