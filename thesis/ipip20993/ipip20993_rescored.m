%% IPIP-NEO-300 (Input Scoring)

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

projectDir = fileparts(fileparts(which(mfilename)));

datasetName = 'ipip20993';

imageDir = fullfile(projectDir, 'images', [datasetName '_rescored']);
tableDir = fullfile(projectDir, 'tables', [datasetName '_rescored']);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

dataFile = fullfile(datasetDir, [datasetName '.dat']);
scoringFile = fullfile(datasetDir, 'IPIP-NEO-300 scoring tool_2.xlsx');

mkdir(imageDir);
mkdir(tableDir);

%% Load Data
opts = detectImportOptions(dataFile);
opts = setvartype(opts, "double");
opts = setvaropts(opts, TreatAsMissing="0");
opts.MissingRule = "omitrow";

data = readtable(dataFile, opts);
data.SEX = categorical(data.SEX, [1 2], {'male', 'female'});
data = rmmissing(data);

opts = detectImportOptions(scoringFile);
scoringTool = readtable(scoringFile, opts);
reverseCoded = cellfun(@(s)s(1), scoringTool.Sign) == '-';
reverseCoded = [false false reverseCoded'];

data(:, reverseCoded) = 6 - data(:, reverseCoded);

clear opts scoringTool reverseCoded;

unlabeledData = removevars(data, 303:width(data));
% transform SEX from [1, 2] to [0, 1]
unlabeledData.SEX = double(unlabeledData.SEX) - 1;
unlabeledData = table2array(unlabeledData);


%% Visualize

Y = tsne(unlabeledData);
fig = figure(1);
scatter(Y(:,1), Y(:,2));
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "e02_tsne.png"),'png');

clear Y;

%% Run

alpha = 0.7;
s = struct();

% kModel = AutoRbfKernel(unlabeledData);
% kModel = DiracKernel();
% kModel = M3Kernel(unlabeledData);
kModel = K1Kernel(unlabeledData);

% encodedData = join(string(unlabeledData), "");
% kModel = StringSubsequenceKernel(maxSubsequence=10, lambda=0.6);

poc = kMRCD(kModel); 
solution = poc.runAlgorithm(unlabeledData, alpha);

alpha = 0.5;

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