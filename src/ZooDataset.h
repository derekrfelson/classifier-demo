/*
 * ZooDataset.h
 *
 *  Created on: Mar 18, 2016
 *      Author: derek
 */

#ifndef ZOODATASET_H_
#define ZOODATASET_H_

#include <string>
#include <eigen3/Eigen/Dense>
#include <memory>
#include "Partition.h"

class ZooDataset
{
public:
	static constexpr auto NumFields = 16;
	static constexpr auto NumClasses = 7;
	using RowVector = Eigen::Matrix<double, 1, NumFields>;
	using CovarianceMatrix = Eigen::Matrix<double, NumFields, NumFields>;

private:
	using TypeVector = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>;
	using DataMatrix = Eigen::Matrix<uint8_t, Eigen::Dynamic, NumFields>;

public:
	explicit ZooDataset(std::string filename);
	size_t size() const;
	RowVector getMeans() const;
	ZooDataset getSubsetByClass(uint8_t type) const;
	Partition<ZooDataset> partition(size_t startIndex, size_t endIndex) const;
	CovarianceMatrix getCovarianceMatrixInverse() const;
	double getCovarianceMatrixDeterminant() const;
	RowVector getPoint(size_t i) const;
	uint8_t getType(size_t i) const;
	std::string getName(size_t i) const;

private:
	explicit ZooDataset(std::vector<std::string> names,
			TypeVector types, DataMatrix data);
	CovarianceMatrix getCovarianceMatrix() const;

	std::vector<std::string> names;
	TypeVector types;
	DataMatrix data;
};

#endif /* ZOODATASET_H_ */
