%% Shakespear vs Trump
% Compute the case study of shakespear vs trump

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

% modelName = 'GIST-small-Embedding-v0';
modelName = 'nomic-embed-text-v1.5';
% modelName = 'all-mpnet-base-v2';
% modelName = 'bge-large-en-v1.5';
% modelName = 'bge-small-en-v1.5';
% modelName = 'all-MiniLM-L6-v2';

datasetName = 'shakespear_trump';
% datasetName = 'shakespear_trump_cleaned';

matryoshkaDim = 64;

imageDir = fullfile(projectDir, 'images', [datasetName '_kernels']);
tableDir = fullfile(projectDir, 'tables', [datasetName '_kernels']);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

datasetFile = fullfile(datasetDir, [datasetName '_' modelName '.parquet']);

folderName = modelName;

if ~isnan(matryoshkaDim)
    folderName = [modelName '-' int2str(matryoshkaDim)];
end

mkdir(imageDir, folderName);
mkdir(tableDir, folderName);

imageDir = fullfile(imageDir, folderName);
tableDir = fullfile(tableDir, folderName);

%% Load Data

[embeddings, labels] = generateSample(datasetFile, 1000, 0.2, matryoshkaDim=matryoshkaDim);

%% Visualize

Y = tsne(embeddings, Distance="cosine");
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "tsne.png"),'png');

clear Y;

%% Run

alpha = 0.7;
s = struct();

%% RBF
kModel = AutoRbfKernel(embeddings);
solution = kMRCD(kModel).runAlgorithm(embeddings, alpha);

s(1).kernel = "AutoRbfKernel Median";
s(1).solution = solution;

%% RBF (Modified mean)
kModel = AutoRbfKernel(embeddings, bandwidth="modifiedmean");
solution = kMRCD(kModel).runAlgorithm(embeddings, alpha);

s(2).kernel = "AutoRbfKernel (Modified Mean)";
s(2).solution = solution;

%% Sphere-RBF
kModel = AutoSphereRbfKernel(embeddings);
solution = kMRCD(kModel).runAlgorithm(embeddings, alpha);

s(3).kernel = "AutoSphereRbfKernel";
s(3).solution = solution;

%% Sphere-RBF
kModel = AutoSphereRbfKernel(embeddingsm, bandwidth="modifiedmean");
solution = kMRCD(kModel).runAlgorithm(embeddings, alpha);

s(4).kernel = "AutoSphereRbfKernel (Modified Mean)";
s(4).solution = solution;

%% String-Subsequence
% kModel = StringSubsequenceKernel(lambda=0.6, maxSubsequence=15);
% solution = kMRCD(kModel).runAlgorithm(text, alpha);

% s(5).kernel = "StringSubsequenceKernel";
% s(5).solution = solution;

%% Summary

for i = 1:numel(s)
    fig = figure();
    
    mahalchart(labels, s(i).solution.rd, s(i).solution.cutoff);
    saveas(fig, fullfile(imageDir, sprintf("mahalanobis_distances_%s.png", s(i).kernel)),'png');
    
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

%% Functions

function [embeddings,labels] = generateSample(filepath, sampleSize, outlierContamination, NameValueArgs)
    arguments
        filepath (1,1) string {mustBeFile}
        sampleSize (1,1) double {mustBeInteger, mustBePositive}
        outlierContamination (1,1) double {mustBeInRange(outlierContamination,0,1)}
        NameValueArgs.matryoshkaDim (1,1) double {mustBePositive, mustBeInteger} = Inf
    end
    
    embeddingVar = "embedding";
    if ~isnan(NameValueArgs.matryoshkaDim) && isfinite(NameValueArgs.matryoshkaDim)
        embeddingVar = ['embedding_' int2str(NameValueArgs.matryoshkaDim)];
    end
    
    rawData = parquetread(filepath, SelectedVariableNames=["text", "author", embeddingVar]);
    rawData.author = categorical(rawData.author);
    
    shakespearIndices = find(rawData.author == "shakespear");
    trumpIndices = find(rawData.author == "trump");
    
    numShakespearSamples = round(outlierContamination * sampleSize);
    numTrumpSamples = sampleSize - numShakespearSamples;
    
    shakespearSampleIndices = datasample(shakespearIndices, numShakespearSamples, Replace=false);
    trumpSampleIndices = datasample(trumpIndices, numTrumpSamples, Replace=false);
    data = rawData(sort(cat(1, shakespearSampleIndices, trumpSampleIndices)),:);
    
    embeddings = cell2mat(cellfun(@transpose,data{:, embeddingVar}, UniformOutput=false));
    labels = renamecats(data.author, {'trump' 'shakespear'}, {'inlier' 'outlier'});
    
    perm = randperm(height(embeddings));
    embeddings = embeddings(perm, :);
    labels = labels(perm, :);
end