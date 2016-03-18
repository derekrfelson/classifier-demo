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
#include <eigen3/Eigen/Dense>

using AnimalMatrix = Eigen::Matrix<uint8_t, 16, 16>;

class Animal
{
public:
	explicit Animal(std::string csvLine);
	uint8_t getType() const;
private:
	std::string name;
	uint8_t type;
	std::array<uint8_t, 16> data;
	std::array<int, 16> means;
	AnimalMatrix covarianceMatrix;
};

#endif /* ANIMAL_H_ */
