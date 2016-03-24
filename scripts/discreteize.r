install.packages("arules")
library(arules)

x <- read.csv("../data/wine.csv", header=FALSE)
for (i in 1:13) { x[,i] = discretize(x[,i], categories=4, labels=c('1', '2', '3', '4')) }
write.table(x, file="../data/wineDiscrete.csv", row.names=FALSE, na="", col.names=FALSE, sep=",")

x <- read.csv("../data/iris.csv", header=FALSE)
for (i in 1:4) { x[,i] = discretize(x[,i], categories=4, labels=c('1', '2', '3', '4')) }
write.table(x, file="../data/irisDiscrete.csv", row.names=FALSE, na="", col.names=FALSE, sep=",")

x <- read.csv("../data/heartDisease.csv", header=FALSE)
for (i in c(1,4,5,8,10)) { x[,i] = discretize(x[,i], categories=4, labels=c('1', '2', '3', '4')) }
write.table(x, file="../data/heartDiseaseDiscrete.csv", row.names=FALSE, na="", col.names=FALSE, sep=",")
