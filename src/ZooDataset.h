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
	explicit ZooDataset(std::string filename);
private:
	using TypeVector = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>;
	using DataMatrix = Eigen::Matrix<uint8_t, Eigen::Dynamic, 16>;
	std::vector<std::string> names;
	TypeVector types;
	DataMatrix data;
};

#endif /* ZOODATASET_H_ */
