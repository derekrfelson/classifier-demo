/*
 * Dataset.h
 *
 *  Created on: Mar 21, 2016
 *      Author: derek
 */

#ifndef DATASET_H_
#define DATASET_H_

#include <string>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <cstdint>
#include "Partition.h"

class Classifier;
enum class ClassifierType : uint8_t;

constexpr auto ZooFields = 16;
constexpr auto ZooClasses = 7;
constexpr auto CpuFields = 7;
constexpr auto CpuClasses = 8;
constexpr auto HeartDiseaseFields = 13;
constexpr auto HeartDiseaseClasses = 5;

class Dataset
{
public:
	const size_t NumFields;
	const size_t NumClasses;

	using Decimal = long double;
	using RowVector = Eigen::Matrix<Decimal, 1, Eigen::Dynamic>;
	using CovarianceMatrix =
			Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic>;
	using TypeVector = Eigen::Matrix<Decimal, Eigen::Dynamic, 1>;
	using DataMatrix = Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic>;

public:
	explicit Dataset(std::vector<std::string> names,
				TypeVector types, DataMatrix data,
				size_t numClasses);
	size_t size() const;
	RowVector getMeans() const;
	Dataset getSubsetByClass(uint8_t type) const;
	Partition<Dataset> partition(size_t startIndex, size_t endIndex) const;
	RowVector getPoint(size_t i) const;
	uint8_t getType(size_t i) const;
	std::string getName(size_t i) const;
	CovarianceMatrix getCovarianceMatrix(ClassifierType type) const;
	Classifier classifier(ClassifierType type) const;

private:
	std::vector<std::string> names;
	TypeVector types;
	DataMatrix data;
};

Dataset::CovarianceMatrix getPseudoInverse(
		const Dataset::CovarianceMatrix& matrix);
Dataset::Decimal getPseudoDeterminant(
		const Dataset::CovarianceMatrix& matrix);
Dataset readZooDataset(std::string filename);
Dataset readCpuDataset(std::string filename);
Dataset readHeartDiseaseDataset(std::string filename);

#endif /* DATASET_H_ */
