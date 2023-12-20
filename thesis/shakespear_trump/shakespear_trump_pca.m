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

imageDir = fullfile(projectDir, 'images', [datasetName '_pca']);
tableDir = fullfile(projectDir, 'tables', [datasetName '_pca']);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

file = fullfile(datasetDir, [datasetName '_' modelName '.parquet']);

mkdir(imageDir, modelName);
mkdir(tableDir, modelName);

%% Load Data

N = 1000;
[eps20, eps20Labels] = generateSample(file, N, 0.2);

%% e=0.2, a=0.7

alpha = 0.7;
data = eps20;
labels = eps20Labels;

pcaRes = robpca(data, 'k', 50, 'kmax', 50, 'alpha', alpha, 'mcd', 0, 'plots', 0);
scores = pcaRes.T;

clear data;

Y = tsne(scores, Distance="cosine");
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, modelName, "e02_tsne.png"),'png');

clear Y;

kModel = AutoRbfKernel(scores);
poc = kMRCD(kModel); 
solution = poc.runAlgorithm(scores, alpha);

% h Subset
hSubset = table(labels(solution.hsubsetIndices), VariableNames="label");
hSubsetSummary = groupcounts(hSubset, "label");
writetable(hSubsetSummary, fullfile(tableDir, modelName, "e02_h_subset.csv"));

clear hSubset hSubsetSummary;

% Confusion Matrix
grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
grouphat(solution.flaggedOutlierIndices) = "outlier";

cm = confusionmat(labels,grouphat);

fig = figure(2);
confusionchart(fig, cm, categories(labels));
saveas(fig, fullfile(imageDir, modelName, 'e02_confusion_matrix.png'),'png');

clear cm grouphat;

% Mahalanobis Distances
fig = figure(3);
mahalchart(labels, solution.rd, solution.cutoff);
saveas(fig, fullfile(imageDir, modelName, 'e02_mahalanobis_distances.png'),'png');

% Comparison
fig = figure(4);
stats = evaluation(scores, labels, alpha, solution, Estimators={'lof' 'iforest'});
saveas(fig, fullfile(imageDir, modelName, 'e02_pr_curve.png'),'png');
writetable(stats, fullfile(tableDir, modelName, "e02_comparison.csv"));

clear stats;

clear solution kModel alpha poc;
clear data labels;
clear scores pcaRes;
%% Comparison

[eps0, eps0Labels] = generateSample(file, N, 0);
[eps10, eps10Labels] = generateSample(file, N, 0.1);
[eps30, eps30Labels] = generateSample(file, N, 0.3);

stats = [
        runComparison(eps0, eps0Labels, 0, 0.5); runComparison(eps0, eps0Labels, 0, 0.75); runComparison(eps0, eps0Labels, 0, 0.9); ...
        runComparison(eps10, eps10Labels, 0.1, 0.5); runComparison(eps10, eps10Labels, 0.1, 0.75); runComparison(eps10, eps10Labels, 0.1, 0.9); ...
        runComparison(eps20, eps20Labels, 0.2, 0.5); runComparison(eps20, eps20Labels, 0.2, 0.75); runComparison(eps20, eps20Labels, 0.2, 0.9); ...
        runComparison(eps30, eps30Labels, 0.3, 0.5); runComparison(eps30, eps30Labels, 0.3, 0.75); runComparison(eps30, eps30Labels, 0.3, 0.9)
        ];

writetable(stats, fullfile(tableDir, modelName, 'comparison.csv'));

%% Functions

function [embeddings,labels] = generateSample(filepath, sampleSize, outlierContamination)
    arguments
        filepath (1,1) string {mustBeFile}
        sampleSize (1,1) double {mustBeInteger, mustBePositive}
        outlierContamination (1,1) double {mustBeInRange(outlierContamination,0,1)}
    end

    rawData = parquetread(filepath, SelectedVariableNames=["text", "author", "embedding"]);
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
    embeddings = embeddings(perm, :);
    labels = labels(perm, :);
end

function stats = runComparison(data, labels, outlierContamination, robustness)
    arguments
        data double {mustBeNonempty}
        labels categorical {mustBeNonempty}
        outlierContamination (1,1) double {mustBeInRange(outlierContamination,0,1)}
        robustness (1,1) double {mustBeInRange(robustness,0.5,1)}
    end
    
    pcaRes = robpca(data, 'k', 50, 'kmax', 50, 'alpha', robustness, 'mcd', 0, 'plots', 0);
    scores = pcaRes.T;

    kModel = AutoRbfKernel(scores);
    poc = kMRCD(kModel); 
    solution = poc.runAlgorithm(scores, robustness);

    stats = evaluation(scores, labels, robustness, solution, Estimators={'lof' 'iforest'});

    names = string(stats.Properties.RowNames);
    e = repmat(outlierContamination,size(names));
    a = repmat(robustness,size(names));
    stats = [table(e,a,names, VariableNames={'e' 'a' 'Name'}), stats];
    stats.Properties.RowNames = {};
end