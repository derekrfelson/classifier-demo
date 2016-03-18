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
	using MeanRowVector = Eigen::Matrix<double, 1, NumFields>;
	using CovarianceMatrix = Eigen::Matrix<double, NumFields, NumFields>;

private:
	using TypeVector = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>;
	using DataMatrix = Eigen::Matrix<uint8_t, Eigen::Dynamic, NumFields>;

public:
	explicit ZooDataset(std::string filename);
	size_t size() const;
	MeanRowVector getMeans() const;
	CovarianceMatrix getCovarianceMatrix() const;
	ZooDataset getSubsetByClass(uint8_t type) const;
	Partition<ZooDataset> partition(size_t startingFold, size_t numFolds);
	Partition<ZooDataset> partition(size_t leaveOutIndex);

private:
	explicit ZooDataset(std::vector<std::string> names,
			TypeVector types, DataMatrix data);

	std::vector<std::string> names;
	TypeVector types;
	DataMatrix data;
};



#endif /* ZOODATASET_H_ */
