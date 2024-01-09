library(rrcov)
library(PRROC)
library(caret)

set.seed(1634256)

datasetName <- "careless_responding_simulation"
projectDir <- getwd()

datasetDir <- file.path(projectDir, 'datasets', datasetName)
tablesDir <- file.path(projectDir, 'tables', datasetName)

# Functions

loadData <- function(distribution, iteration) {
  datasetFile <-
    file.path(datasetDir, sprintf("dat_%s", distribution), sprintf("dat_%d_%s.csv", iteration, distribution))

  d <- read.csv2(datasetFile, header = TRUE)

  d$Careless <- factor(d$Careless, levels = c(0, 1), labels=c("regular", "careless"))

  unlabeledData <- subset(d, select = -c(Careless))
  labels <- factor(d$Careless, levels = c("regular", "careless"), labels=c("inlier", "outlier"))

  perm <- sample(nrow(d), replace=FALSE)
  unlabeledData <- unlabeledData[perm,]
  labels <- labels[perm]

  return(list(unlabeledData = unlabeledData, labels = labels))
}

runSimulation <- function(distribution, alpha) {
  results = data.frame()

  for (i in 1:100) {
    d <- loadData(distribution, i)

    unlabeledData <- d$unlabeledData
    labels <- d$labels

    res <- CovMrcd(unlabeledData, alpha = alpha)
    flaggedOutliers1 <- !getFlag(res)

    grouphat <- factor(getFlag(res), levels = c(TRUE, FALSE), labels=c("inlier", "outlier"))

    pr <- pr.curve(scores.class0 = res$mah, weights.class0 = as.numeric(labels) - 1)
    cm <- confusionMatrix(data=grouphat, reference = labels, positive = "outlier")

    results = rbind(
      results,
      data.frame(
        accuracy = cm$overall["Accuracy"],
        precision = cm$byClass["Precision"],
        sensitivity = cm$byClass["Sensitivity"],
        specificity = cm$byClass["Specificity"],
        f1Score = cm$byClass["F1"],
        aucpr = pr$auc.integral,
        name = "MRCD",
        iteration = i
      )
    )
  }

  return(results)
}

# Uniform
alpha <- 0.7
results = runSimulation("uni", alpha)
write.csv(results, file.path(tablesDir, sprintf("simulation_uni_a0%.0f_mrcd.csv", alpha*10)), row.names=FALSE)


# Middle
alpha <- 0.7
results = runSimulation("mid", alpha)
write.csv(results, file.path(tablesDir, sprintf("simulation_mid_a0%.0f_mrcd.csv", alpha*10)), row.names=FALSE)

# Pattern
alpha <- 0.7
results = runSimulation("pattern", alpha)
write.csv(results, file.path(tablesDir, sprintf("simulation_pattern_a0%.0f_mrcd.csv", alpha*10)), row.names=FALSE)
