%% Wisconsin Breastcancer (Kernel comparison)

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'breast-cancer-wisconsin';

imageDir = fullfile(projectDir, 'images', [datasetName '_kernels']);
tableDir = fullfile(projectDir, 'tables', [datasetName '_kernels']);
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
legend(Location="southwest");
hold off;
saveas(fig,fullfile(imageDir, "pr_curve.png"),'png');

% Comparison
stats = struct2table(s);
writetable(stats, fullfile(tableDir, "comparison.csv"));

clear s;