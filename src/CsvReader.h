/*
 * CsvReader.h
 *
 *  Created on: Mar 25, 2016
 *      Author: derek
 */

#ifndef CSVREADER_H_
#define CSVREADER_H_

#include "Types.h"
#include <string>
class Dataset;

Dataset readIrisDataset(std::string filename);
Dataset readWineDataset(std::string filename);
Dataset readHeartDiseaseDataset(std::string filename);

#endif /* CSVREADER_H_ */
