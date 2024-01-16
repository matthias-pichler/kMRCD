library(rrcov)
library(PRROC)
library(caret)

set.seed(1634256)

datasetName <- "breast-cancer-wisconsin"
projectDir <- getwd()

imageDir <- file.path(projectDir, 'images', datasetName)
tableDir <- file.path(projectDir, 'tables', datasetName)
datasetDir <- file.path(projectDir, 'datasets', datasetName)

datasetFile <-
  file.path(datasetDir, sprintf("%s.data", datasetName))

data <- read.csv(datasetFile, header = FALSE)
colnames(data) <-
  c(
    "SampleCodeNumber",
    "ClumpThickness",
    "UniformityOfCellSize",
    "UniformityOfCellShape",
    "MarginalAdhesion",
    "SingleEpithelialCellSize",
    "BareNuclei",
    "BlandChromatin",
    "NormalNucleoli",
    "Mitoses",
    "Class"
  )
data$Class <- factor(data$Class, levels=c(2,4), labels = c("benign", "malignant"))

unlabeledData <- subset(data, select = -c(SampleCodeNumber, Class))
labels <- factor(data$Class, levels=c("benign", "malignant"), labels=c("inlier", "outlier"))

perm <- sample(nrow(data), replace=FALSE)
unlabeledData <- unlabeledData[perm,]
labels <- labels[perm]

res1 <- CovMrcd(unlabeledData, alpha = 0.5)
flaggedOutliers1 <- !getFlag(res1)

res2 <- CovMrcd(unlabeledData, alpha = 0.5, rho = 0.1)
grouphat2 <- factor(getFlag(res2), levels = c(TRUE, FALSE), labels=c("inlier", "outlier"))

pr <- pr.curve(scores.class0 = res2$mah, weights.class0 = as.numeric(labels) - 1)
cm <- confusionMatrix(data=grouphat2, reference = labels, positive = "outlier")
