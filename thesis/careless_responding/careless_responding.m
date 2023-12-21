%% Careless Responding

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'careless_responding';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

file = fullfile(datasetDir, 'data_mod_resp.csv');

mkdir(imageDir);
mkdir(tableDir);

%% Load Data
opts = detectImportOptions(file);
opts = setvartype(opts, 'double');

data = readtable(file, opts);
data.Careless = categorical(data.Careless, [0 1], {'regular', 'careless'});

unlabeledData = table2array(removevars(data,{'Var1', 'Careless'}));
labels = renamecats(data.Careless, {'regular' 'careless'}, {'inlier' 'outlier'});

perm = randperm(height(unlabeledData));
unlabeledData = unlabeledData(perm, :);
labels = labels(perm, :);

clear opts perm;

%% Visualize

alpha = 0.5;

Y = tsne(unlabeledData);
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "e02_tsne.png"),'png');

clear Y;

%% Run

% kModel = AutoRbfKernel(unlabeledData);
% kModel = DiracKernel();
% kModel = M3Kernel(unlabeledData);
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