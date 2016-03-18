/*
 * Animal.h
 *
 *  Created on: Mar 17, 2016
 *      Author: derek
 */

#ifndef ANIMAL_H_
#define ANIMAL_H_

#include <string>

struct Animal
{
public:
	explicit Animal(std::string csvLine);
	std::string name;
	bool hair;
	bool feathers;
	bool eggs;
	bool milk;
	bool airborne;
	bool aquatic;
	bool predator;
	bool toothed;
	bool backbone;
	bool breathes;
	bool venomous;
	bool fins;
	unsigned int legs;
	bool tail;
	bool domestic;
	bool catsize;
	unsigned int type;
};

#endif /* ANIMAL_H_ */
