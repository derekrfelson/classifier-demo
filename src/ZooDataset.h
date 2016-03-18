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

class ZooDataset
{
public:
	using MeanRowVector = Eigen::Matrix<double, 1, 16>;
	using CovarianceMatrix = Eigen::Matrix<double, 16, 16>;

	explicit ZooDataset(std::string filename);
	MeanRowVector getMeans() const;
	CovarianceMatrix getCovarianceMatrix() const;
private:
	using TypeVector = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>;
	using DataMatrix = Eigen::Matrix<uint8_t, Eigen::Dynamic, 16>;

	std::vector<std::string> names;
	TypeVector types;
	DataMatrix data;
};

#endif /* ZOODATASET_H_ */
