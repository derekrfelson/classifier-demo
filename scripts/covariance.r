x <- read.csv("../data/wine.csv", header=FALSE)
write.table(cov(x[,1:13]), file="../data/wineCovariance.csv", row.names=FALSE, quote=FALSE, na="", col.names=FALSE, sep=",")
x <- read.csv("../data/wineDiscrete.csv", header=FALSE)
write.table(cov(x[,1:13]), file="../data/wineDiscreteCovariance.csv", row.names=FALSE, quote=FALSE, na="", col.names=FALSE, sep=",")

x <- read.csv("../data/iris.csv", header=FALSE)
write.table(cov(x[,1:4]), file="../data/irisCovariance.csv", row.names=FALSE, quote=FALSE, na="", col.names=FALSE, sep=",")
x <- read.csv("../data/irisDiscrete.csv", header=FALSE)
write.table(cov(x[,1:4]), file="../data/irisDiscreteCovariance.csv", row.names=FALSE, quote=FALSE, na="", col.names=FALSE, sep=",")

x <- read.csv("../data/heartDisease.csv", header=FALSE)
write.table(cov(x[,1:13]), file="../data/heartDiseaseCovariance.csv", row.names=FALSE, quote=FALSE, na="", col.names=FALSE, sep=",")
x <- read.csv("../data/heartDiseaseDiscrete.csv", header=FALSE)
write.table(cov(x[,1:13]), file="../data/heartDiseaseDiscreteCovariance.csv", row.names=FALSE, quote=FALSE, na="", col.names=FALSE, sep=",")

