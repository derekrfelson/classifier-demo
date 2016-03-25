/*
 * Types.h
 *
 *  Created on: Mar 25, 2016
 *      Author: derek
 */

#ifndef TYPES_H_
#define TYPES_H_

#include <cstdint>
#include <eigen3/Eigen/Dense>

using Decimal = long double;
using RowVector = Eigen::Matrix<Decimal, 1, Eigen::Dynamic>;
using CovarianceMatrix =
	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic>;
using TypeVector = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>;
using ColVector = Eigen::Matrix<Decimal, Eigen::Dynamic, 1>;
using DataMatrix = Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic>;

enum class ClassifierType : uint8_t
{
	OPTIMAL,
	NAIVE,
	LINEAR,
	DECISION_TREE
};

constexpr auto WineFields = 13;
constexpr auto WineClasses = 3;
constexpr auto IrisFields = 4;
constexpr auto IrisClasses = 3;
constexpr auto HeartDiseaseFields = 13;
constexpr auto HeartDiseaseClasses = 4;

#endif /* TYPES_H_ */
