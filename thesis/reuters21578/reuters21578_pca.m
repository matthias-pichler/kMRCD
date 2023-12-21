%% Reuters 21578
% See https://faculty.cc.gatech.edu/~hpark/papers/textoutlier17.pdf

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

modelName = 'all-mpnet-base-v2';
% modelName = 'bge-large-en-v1.5';
% modelName = 'all-MiniLM-L6-v2';
% modelName = 'bge-small-en-v1.5';

datasetName = 'reuters21578';
% datasetName = 'reuters21578_cleaned';

imageDir = fullfile(projectDir, 'images', [datasetName '_pca']);
tableDir = fullfile(projectDir, 'tables', [datasetName '_pca']);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

file = fullfile(datasetDir, [datasetName '_' modelName '.parquet']);

mkdir(imageDir, modelName);
mkdir(tableDir, modelName);

%% Load Data

rawData = parquetread(file, SelectedVariableNames=["text", "topics", "embedding"]);
rawData.topics = cellfun(@categorical, rawData.topics, UniformOutput=false);

inlierTopics = {'acq', 'earn'};

inlierIndices = find(cellfun(@(t)any(ismember(inlierTopics, t)), rawData.topics));
outlierIndices = find(cellfun(@(t)ismember("interest", t), rawData.topics));

%inlierIndices = datasample(inlierIndices, 900, Replace=false);
%outlierIndices = datasample(outlierIndices, 100, Replace=false);

data = rawData(sort(cat(1, inlierIndices, outlierIndices)),:);
outlierIndices = cellfun(@(t)ismember("interest", t), data.topics);
    
embeddings = cell2mat(cellfun(@transpose,data.embedding, UniformOutput=false));
labels = categorical(repmat("inlier", height(data), 1), {'inlier' 'outlier'});
labels(outlierIndices) = "outlier";

perm = randperm(height(data));
embeddings = embeddings(perm, :);
labels = labels(perm, :);

clear rawData inlierIndices outlierIndices perm;
%% PCA

alpha = 0.7;

pcaRes = robpca(embeddings, 'k', 50, 'kmax', 50, 'alpha', alpha, 'mcd', 0, 'plots', 0);
scores = pcaRes.T;

clear pcaRes embeddings;

%% Visualize

Y = tsne(scores, Distance="cosine");
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, modelName, "tsne.png"),'png');

clear Y;

%% Run

kModel = AutoRbfKernel(scores);
poc = kMRCD(kModel); 
solution = poc.runAlgorithm(scores, alpha);

%% Evaluation

% h Subset
hSubset = table(labels(solution.hsubsetIndices), VariableNames="label");
hSubsetSummary = groupcounts(hSubset, "label");
writetable(hSubsetSummary, fullfile(tableDir, modelName, "e02_h_subset.csv"));

clear hSubset hSubsetSummary;

% Confusion Matrix
grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
grouphat(solution.flaggedOutlierIndices) = "outlier";

cm = confusionmat(labels,grouphat);

fig = figure(2);
confusionchart(fig, cm, categories(labels));
saveas(fig, fullfile(imageDir, modelName, 'confusion_matrix.png'),'png');

clear cm grouphat;

% Mahalanobis Distances
fig = figure(3);
mahalchart(labels, solution.rd, solution.cutoff);
saveas(fig, fullfile(imageDir, modelName, 'mahalanobis_distances.png'),'png');

% Comparison
fig = figure(4);
stats = evaluation(scores, labels, alpha, solution);
saveas(fig, fullfile(imageDir, modelName, 'pr_curve.png'),'png');
writetable(stats, fullfile(tableDir, modelName, "comparison.csv"));

clear stats;

clear solution kModel alpha poc;
clear data labels embeddings scores;
    