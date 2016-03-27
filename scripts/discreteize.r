#install.packages("arules")
library(arules)

numcols <- 3;
clabels <- c(1, 2, 3)

x <- read.csv("../data/wine.csv", header=FALSE)
for (i in 1:13) { x[,i] = discretize(x[,i], categories=numcols, labels=clabels) }
write.table(x, file="../data/wineDiscrete.csv", row.names=FALSE, quote=FALSE, na="", col.names=FALSE, sep=",")

x <- read.csv("../data/iris.csv", header=FALSE)
for (i in 1:4) { x[,i] = discretize(x[,i], categories=numcols, labels=clabels) }
write.table(x, file="../data/irisDiscrete.csv", row.names=FALSE, quote=FALSE, na="", col.names=FALSE, sep=",")

x <- read.csv("../data/heartDisease.csv", header=FALSE)
for (i in c(1,4,5,8,10)) { x[,i] = discretize(x[,i], categories=numcols, labels=clabels) }
write.table(x, file="../data/heartDiseaseDiscrete.csv", row.names=FALSE, quote=FALSE, na="", col.names=FALSE, sep=",")
