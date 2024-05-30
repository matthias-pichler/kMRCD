%% Lymphography

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'lymphography';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

file = fullfile(datasetDir, [datasetName '.data']);

mkdir(imageDir);
mkdir(tableDir);

%% Load Data
opts = delimitedTextImportOptions(NumVariables=19, DataLines=[1, Inf], ...
    Delimiter=",", ExtraColumnsRule="ignore", EmptyLineRule="skip", ImportErrorRule="omitrow",...
    VariableNames=["class", "lymphatics", "block_of_affere", "bl_of_lymph_c", ...
        "bl_of_lymph_s", "by_pass", "extravasates", "regeneration_of", "early_uptake_in", ...
        "lym_nodes_dimin", "lym_nodes_enlar", "changes_in_lym", "defect_in_node",...
        "changes_in_node", "changes_in_stru", "special_forms", "dislocation_of"...
        "exclusion_of_no", "no_of_nodes_in"], ...
    VariableTypes = ["categorical", "double", "double", "double", "double", ...
        "double", "double", "double", "double", "double", "double", "double", ...
        "double", "double", "double", "double", "double", "double", "double" ]);

data = readtable(file, opts);
data.class = renamecats(data.class, {'1' '2' '3' '4'}, {'normal find' 'metastases' 'malign lymph' 'fibrosis'});

clear opts;

unlabeledData = table2array(removevars(data,{'class'}));
labels = mergecats(data.class, {'normal find' 'fibrosis'}, 'outlier');
labels = mergecats(labels, {'metastases' 'malign lymph'}, 'inlier');

perm = randperm(height(unlabeledData));
unlabeledData = unlabeledData(perm, :);
labels = labels(perm, :);

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

poc = kMRCD(kModel, cutoffEstimator='skewnessAdjusted'); 
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

clear stats;

clear solution kModel alpha poc;