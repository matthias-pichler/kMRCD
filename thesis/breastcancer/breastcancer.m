%% Wisconsin Breastcancer

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'breast-cancer-wisconsin';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

file = fullfile(datasetDir, [datasetName '.data']);

mkdir(imageDir);
mkdir(tableDir);

%% Load Data
opts = delimitedTextImportOptions(NumVariables=11,DataLines=[1, Inf], ...
    Delimiter=",", ExtraColumnsRule="ignore", EmptyLineRule="skip", ImportErrorRule="omitrow",...
    VariableNames=["SampleCodeNumber", "ClumpThickness", "UniformityOfCellSize", ...
    "UniformityOfCellShape", "MarginalAdhesion", "SingleEpithelialCellSize", ...
    "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "Class"], ...
    VariableTypes = ["int32", "double", "double", "double", "double", ...
    "double", "double", "double", "double", "double", "categorical"]);

data = readtable(file, opts);
data.Class = renamecats(data.Class, {'2' '4'}, {'benign', 'malignant'});

unlabeledData = table2array(removevars(data,{'SampleCodeNumber', 'Class'}));
labels = renamecats(data.Class, {'benign' 'malignant'}, {'inlier' 'outlier'});

perm = randperm(height(unlabeledData));
unlabeledData = unlabeledData(perm, :);
labels = labels(perm, :);

clear opts perm;

%% Run

alpha = 0.7;

Y = tsne(unlabeledData);
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "e02_tsne.png"),'png');

clear Y;

% kModel = AutoRbfKernel(unlabeledData);
% kModel = DiracKernel();
% kModel = M3Kernel(unlabeledData);
kModel = K1Kernel(unlabeledData);

poc = kMRCD(kModel, cutoffEstimator='skewedbox');
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
stats = evaluation(unlabeledData, labels, alpha, solution, CategoricalPredictors="all");
saveas(fig, fullfile(imageDir, 'pr_curve.png'),'png');
writetable(stats, fullfile(tableDir, "comparison.csv"));
