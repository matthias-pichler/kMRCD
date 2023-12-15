%% Discretized Normal Distribution

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

projectDir = fileparts(fileparts(which(mfilename)));

datasetName = 'normal_discretized_kernels';

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

% Comparison
stats = struct2table(s);
writetable(stats, fullfile(tableDir, "comparison.csv"));

clear s;