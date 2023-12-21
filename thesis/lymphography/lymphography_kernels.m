%% Lymphography

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'lymphography';

imageDir = fullfile(projectDir, 'images', [datasetName '_kernels']);
tableDir = fullfile(projectDir, 'tables', [datasetName '_kernels']);
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
rbfModel = LinKernel();
solution = kMRCD(rbfModel).runAlgorithm(unlabeledData, alpha);

s(1).kernel = "Linear";
s(1).solution = solution;

%% RBF
rbfModel = AutoRbfKernel(unlabeledData);
solution = kMRCD(rbfModel).runAlgorithm(unlabeledData, alpha);

s(2).kernel = "RBF";
s(2).solution = solution;

%% Dirac

diracModel = DiracKernel();
solution = kMRCD(diracModel).runAlgorithm(unlabeledData, alpha);
s(3).kernel = "Dirac";
s(3).solution = solution;

%% k1
k1Model = K1Kernel(unlabeledData);
solution = kMRCD(k1Model).runAlgorithm(unlabeledData, alpha);
s(4).kernel = "k1";
s(4).solution = solution;

%% m3
m3Model = M3Kernel(unlabeledData);
solution = kMRCD(m3Model).runAlgorithm(unlabeledData, alpha);
s(5).kernel = "m3";
s(5).solution = solution;

%% Aitchison-Aitken
aaModel = AitchisonAitkenKernel(unlabeledData);
solution = kMRCD(aaModel).runAlgorithm(unlabeledData, alpha);
s(6).kernel = "Aitchison-Aitken";
s(6).solution = solution;

%% Li-Racin
lrModel = LiRacinKernel(unlabeledData);
solution = kMRCD(lrModel).runAlgorithm(unlabeledData, alpha);
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
saveas(fig,fullfile(imageDir, "pr_curve.png"),'png');

% Comparison
stats = struct2table(s);
writetable(stats, fullfile(tableDir, "comparison.csv"));

clear s;