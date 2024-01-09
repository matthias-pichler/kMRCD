library(rrcov)
library(PRROC)
library(caret)

set.seed(1634256)

datasetName <- "normal_discretized"
projectDir <- getwd()

datasetDir <- file.path(projectDir, 'datasets', datasetName)
tablesDir <- file.path(projectDir, 'tables', datasetName)

# Run 1
categories <- 5
dimensions <- 30
contamination <- 0.2
alpha <- 0.5

start <- 1
iter <- 100

results = data.frame()

for (i in start:iter) {
  datasetFile <-
    file.path(datasetDir, sprintf("data_c%d_d%d_e0%.0f", categories, dimensions, contamination*10), sprintf("data_%d.csv", i))
  data <- read.csv(datasetFile, header = TRUE)
  data$labels <- as.factor(data$labels)

  unlabeledData <- subset(data, select = -c(labels))
  labels <- data$labels

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

write.csv(results, file.path(tablesDir, sprintf("simulation_a0%.0f_e0%.0f_mrcd.csv", alpha*10, contamination*10)), row.names=FALSE)
