%% Careless Responding (Kernel comparison)

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'careless_responding';

imageDir = fullfile(projectDir, 'images', [datasetName '_kernels']);
tableDir = fullfile(projectDir, 'tables', [datasetName '_kernels']);
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
encodedData = join(string(unlabeledData), "");

clear opts perm;

%% Visualize

Y = tsne(unlabeledData);
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "e02_tsne.png"),'png');

clear Y;

%% Run

alpha = 0.7;
s = struct();

%% Linear

kModel = LinKernel();
solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
s(1).kernel = "Linear";
s(1).solution = solution;

%% RBF

kModel = AutoRbfKernel(unlabeledData);
solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
s(2).kernel = "RBF";
s(2).solution = solution;

%% Dirac

kModel = DiracKernel();
solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
s(3).kernel = "Dirac";
s(3).solution = solution;

%% k1

kModel = K1Kernel(unlabeledData);
solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
s(4).kernel = "k1";
s(4).solution = solution;

%% m3

kModel = M3Kernel(unlabeledData);
solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
s(5).kernel = "m3";
s(5).solution = solution;

%% Aitchison-Aitken

kModel = AitchisonAitkenKernel(unlabeledData);
solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
s(6).kernel = "Aitchison-Aitken";
s(6).solution = solution;

%% Li-Racin

kModel = LiRacinKernel(unlabeledData);
solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
s(7).kernel = "Li-Racin";
s(7).solution = solution;

%% String-Subsequence

kModel = StringSubsequenceKernel(maxSubsequence=10, lambda=0.1);
solution = kMRCD(kModel).runAlgorithm(encodedData, alpha);
s(8).kernel = "SSK";
s(8).solution = solution;

%% Ordered Aitchison-Aitken

kModel = OrderedAitchisonAitkenKernel(unlabeledData);
solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
s(9).kernel = "Ordered Aitchison-Aitken";
s(9).solution = solution;

%% Ordered Li-Racin

kModel = OrderedLiRacinKernel(unlabeledData, lambda=repmat(0.01, 1, width(unlabeledData)));
solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
s(10).kernel = "Ordered Li-Racin";
s(10).solution = solution;

%% Wang-Ryzin

kModel = WangRyzinKernel(unlabeledData, lambda=repmat(0.01, 1, width(unlabeledData)));
solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
s(11).kernel = "Wang-Ryzin";
s(11).solution = solution;

%% Summary

for i = 1:numel(s)
    grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
    grouphat(s(i).solution.flaggedOutlierIndices) = "outlier";
    cm = confusionmat(labels,grouphat);
    stats = confusionstats(confusionmat(labels,grouphat, Order={'outlier' 'inlier'}));

    s(i).accuracy = stats.accuracy;
    s(i).precision = stats.precision;
    s(i).sensitivity = stats.sensitivity;
    s(i).specificity = stats.specificity;
    s(i).f1Score = stats.f1Score;
end

%% PR-Curve

outlierRatio = sum(labels == "outlier")/numel(labels);
colors = jet(numel(s));

fig = figure(2);
hold on;

xlabel("Recall");
ylabel("Precision");
title("Precision-Recall Curve");

for i = 1:numel(s)
    scores = s(i).solution.rd;
    auc = prcurve(labels,scores,'outlier',DisplayName=s(i).kernel, Color=colors(i,:));
    s(i).aucpr = auc;
end

yline(outlierRatio, LineStyle="--", ...
      DisplayName=sprintf("No Skill Classifier (AUC=%0.4f)", outlierRatio));
legend;
hold off;
saveas(fig,fullfile(imageDir, "pr_curve.png"),'png');

% Comparison
stats = struct2table(s);
writetable(stats, fullfile(tableDir, "comparison.csv"));