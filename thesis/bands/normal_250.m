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
p = 200;
eps = 0;

ndm = ALYZ.NewDataModel(ALYZ.ALYZCorrelationType(), ALYZ.ClusterContamination());
[unlabeledData, ~, ~,idxOutliers] = ndm.generateDataset(N, p, eps, 20);

% unlabeledData = rZscores(unlabeledData); Not solved by centering

labels = categorical(repmat("inlier", [N 1]), {'inlier' 'outlier'});
labels(idxOutliers) = "outlier";

x = unlabeledData;
% x = rZscores(unlabeledData);

% kModel = AutoRbfKernel(x);
% kModel = PolyKernel(3);
kModel = LinKernel();

poc = kMRCD(kModel, cutoffEstimator="lognormal");
solution = poc.runAlgorithm(x, alpha);

% Mahalanobis Distances
figure();
hSubset = categorical(repmat("not in H", [N 1]), {'in H' 'not in H'});
hSubset(solution.hsubsetIndices) = "in H";
mahalchart(hSubset, solution.rd, solution.cutoff);

%% Matrix

K = kModel.compute(x);

s = svd(Utils.center(kModel.compute(x(solution.hsubsetIndices, :))));
e_min = min(s);
e_max = max(s);

Ktt_diag = diag(K);
Kx = kModel.compute(x(solution.hsubsetIndices, :));
nx = size(Kx,1);
Kt = kModel.compute(x, x(solution.hsubsetIndices, :));
Kc = Utils.center(Kx);
Kt_c = Utils.center(Kx,Kt);
Kxx = Ktt_diag - (2/nx)*sum(Kt,2) + (1/nx^2)*sum(sum(Kx));

%% Comparison

[~, ~ , mah, idxOutliers] = robustcov(unlabeledData, OutlierFraction=(1-alpha));

labels = categorical(repmat("inlier", [N 1]), {'inlier' 'outlier'});
labels(idxOutliers) = "outlier";

cutoff = sqrt(chi2inv(0.975,p));

figure();
mahalchart(labels, mah, cutoff);
