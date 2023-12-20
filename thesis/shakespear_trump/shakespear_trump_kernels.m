%% Shakespear vs Trump
% Compute the case study of shakespear vs trump

% Setup
clc;
clear all;
close all;

rng(1634256, "twister");

projectDir = fileparts(fileparts(which(mfilename)));

modelName = 'all-mpnet-base-v2';
% modelName = 'bge-large-en-v1.5';
% modelName = 'bge-small-en-v1.5';
% modelName = 'all-MiniLM-L6-v2';

datasetName = 'shakespear_trump';
% datasetName = 'shakespear_trump_cleaned';

imageDir = fullfile(projectDir, 'images', [datasetName '_kernels']);
tableDir = fullfile(projectDir, 'tables', [datasetName '_kernels']);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

file = fullfile(datasetDir, [datasetName '_' modelName '.parquet']);

mkdir(imageDir, modelName);
mkdir(tableDir, modelName);

%% Load Data

sampleSize = 1000;
outlierContamination= 0.2;

rawData = parquetread(file, SelectedVariableNames=["text", "author", "embedding"]);
rawData.author = categorical(rawData.author);

shakespearIndices = find(rawData.author == "shakespear");
trumpIndices = find(rawData.author == "trump");

numShakespearSamples = round(outlierContamination * sampleSize);
numTrumpSamples = sampleSize - numShakespearSamples;

shakespearSampleIndices = datasample(shakespearIndices, numShakespearSamples, Replace=false);
trumpSampleIndices = datasample(trumpIndices, numTrumpSamples, Replace=false);
data = rawData(sort(cat(1, shakespearSampleIndices, trumpSampleIndices)),:);

embeddings = cell2mat(cellfun(@transpose,data.embedding, UniformOutput=false));
labels = renamecats(data.author, {'trump' 'shakespear'}, {'inlier' 'outlier'});

perm = randperm(height(embeddings));
text = data.text(perm, :);
embeddings = embeddings(perm, :);
labels = labels(perm, :);

clear numShakespearSamples numTrumpSamples;
clear shakespearSampleIndices trumpSampleIndices;
clear rawData perm;

%% Visualize

Y = tsne(data, Distance="cosine");
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, modelName, "e02_tsne.png"),'png');

clear Y;

%% Run

alpha = 0.7;
s = struct();

%% RBF
kModel = AutoRbfKernel(embeddings);
solution = kMRCD(kModel).runAlgorithm(embeddings, alpha);

s(1).kernel = "RBF";
s(1).solution = solution;

%% Sphere-RBF
kModel = AutoSphereRbfKernel(embeddings);
solution = kMRCD(kModel).runAlgorithm(embeddings, alpha);

s(2).kernel = "Sphere-RBF";
s(2).solution = solution;

%% String-Subsequence
kModel = StringSubsequenceKernel(text);
solution = kMRCD(kModel).runAlgorithm(text, alpha);

s(3).kernel = "SSK";
s(3).solution = solution;

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