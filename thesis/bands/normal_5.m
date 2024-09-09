%% Normal Distribution

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'bands';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

mkdir(imageDir);
mkdir(tableDir);

%% Run

N = 1000;
alpha = 0.7;
p = 5;
eps = 0;

ndm = ALYZ.NewDataModel(ALYZ.ALYZCorrelationType(), ALYZ.ClusterContamination());
[unlabeledData, ~, ~,idxOutliers] = ndm.generateDataset(N, p, eps, 20);

unlabeledData = rZscores(unlabeledData);

labels = categorical(repmat("inlier", [N 1]), {'inlier' 'outlier'});
labels(idxOutliers) = "outlier";

kModel = AutoRbfKernel(unlabeledData);

poc = kMRCD(kModel);
solution = poc.runAlgorithm(unlabeledData, alpha);

% Mahalanobis Distances
figure();
hSubset = categorical(repmat("not in H", [N 1]), {'in H' 'not in H'});
hSubset(solution.hsubsetIndices) = "in H";
mahalchart(hSubset, solution.rd, solution.cutoff);

%% Comparison

[~, ~ , mah, idxOutliers, s] = robustcov(unlabeledData, OutlierFraction=(1-alpha));

labels = categorical(repmat("inlier", [N 1]), {'inlier' 'outlier'});
labels(idxOutliers) = "outlier";

cutoff = sqrt(chi2inv(0.975,p));

figure();
mahalchart(labels, mah, cutoff);
