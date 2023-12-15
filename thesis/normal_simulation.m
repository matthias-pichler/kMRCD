%% Discretized Normal Distribution

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

projectDir = fileparts(fileparts(which(mfilename)));

datasetName = 'normal_discretized';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);

mkdir(imageDir);
mkdir(tableDir);

%% Generate Data

contamination = 0.2;
numCategories = 5;
dimensions = 32;
N = 1000;

ndm = NewDataModel(ALYZCorrelationType(), ClusterContamination());
[x, ~, ~,idxOutliers] = ndm.generateDataset(N, dimensions, contamination, 20);        

unlabeledData = cell2mat(cellfun(@(X)discretize(X, numCategories), num2cell(x, 1), UniformOutput=false));

labels = categorical(repmat("inlier", [N 1]), {'inlier' 'outlier'});
labels(idxOutliers) = "outlier";

clear x;

%% Visualize

Y = tsne(unlabeledData);
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "e02_tsne.png"),'png');

clear Y;

%% Run

alpha = 0.7;

% kModel = AutoRbfKernel(data);
% kModel = DiracKernel();
% kModel = M3Kernel(data);
kModel = K1Kernel(unlabeledData);

poc = kMRCD(kModel); 
solution = poc.runAlgorithm(unlabeledData, alpha);

% h Subset
hSubset = table(labels(solution.hsubsetIndices), VariableNames="label");
hSubsetSummary = groupcounts(hSubset, "label");
writetable(hSubsetSummary, fullfile(tableDir, "h_subset.csv"));

clear hSubset hSubsetSummary;

% Confusion Matrix
grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
grouphat(solution.flaggedOutlierIndices) = "outlier";

cm = confusionmat(labels,grouphat);

fig = figure(2);
confusionchart(fig, cm, categories(labels));
saveas(fig, fullfile(imageDir, 'confusion_matrix.png'),'png');

clear cm grouphat;

% Mahalanobis Distances
fig = figure(3);
mahalchart(labels, solution.rd, solution.cutoff);
saveas(fig, fullfile(imageDir, 'mahalanobis_distances.png'),'png');

% Comparison
fig = figure(4);
stats = evaluation(unlabeledData, labels, alpha, solution);
saveas(fig, fullfile(imageDir, 'pr_curve.png'),'png');
writetable(stats, fullfile(tableDir, "comparison.csv"));

clear stats;

clear solution kModel alpha poc;