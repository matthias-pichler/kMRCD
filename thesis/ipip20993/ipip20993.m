%% IPIP-NEO-300

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'ipip20993';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

file = fullfile(datasetDir, [datasetName '.dat']);

mkdir(imageDir);
mkdir(tableDir);

%% Load Data
opts = detectImportOptions(file);
opts = setvartype(opts, "double");
opts = setvaropts(opts, TreatAsMissing="0");
opts.MissingRule = "omitrow";

data = readtable(file, opts);
data.SEX = categorical(data.SEX, [1 2], {'male', 'female'});
data = rmmissing(data);

unlabeledData = removevars(data, 303:width(data));
% transform SEX from [1, 2] to [0, 1]
unlabeledData.SEX = double(unlabeledData.SEX) - 1;
unlabeledData = table2array(unlabeledData);

clear opts;

%% Visualize

alpha = 0.5;

Y = tsne(unlabeledData);
fig = figure(1);
scatter(Y(:,1), Y(:,2));
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "e02_tsne.png"),'png');

clear Y;

%% Run
% kModel = AutoRbfKernel(unlabeledData);
% kModel = DiracKernel();
% kModel = M3Kernel(unlabeledData);
kModel = K1Kernel(unlabeledData);

% encodedData = join(string(unlabeledData), "");
% kModel = StringSubsequenceKernel(maxSubsequence=10, lambda=0.6);

poc = kMRCD(kModel); 
solution = poc.runAlgorithm(unlabeledData, alpha);

% placeholder
labels = categorical(repmat("inlier", [height(unlabeledData), 1]), {'inlier' 'outlier'});

grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
grouphat(solution.flaggedOutlierIndices) = "outlier";

outliers = data(solution.flaggedOutlierIndices, :);

clear grouphat;

% Mahalanobis Distances
fig = figure(3);
mahalchart(labels, solution.rd, solution.cutoff);
saveas(fig, fullfile(imageDir, 'mahalanobis_distances.png'),'png');

clear solution kModel alpha poc;