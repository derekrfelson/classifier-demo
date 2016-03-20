/*
 * Partition.h
 *
 *  Created on: Mar 18, 2016
 *      Author: derek
 */

#ifndef PARTITION_H_
#define PARTITION_H_

#include <cstddef>
#include <utility>

std::pair<size_t, size_t> kFoldIndices(size_t currentFold, size_t k, size_t size);

template <typename T>
struct Partition
{
public:
	Partition(const T& training, const T& testing);
	T training;
	T testing;
};

template <typename T>
Partition<T>::Partition(const T& training, const T& testing)
: training{training},
  testing{testing}
{
}

#endif /* PARTITION_H_ */
