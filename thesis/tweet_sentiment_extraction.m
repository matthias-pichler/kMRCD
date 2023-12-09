%% Tweet Sentiment
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

% datasetName = 'tweet_sentiment_extraction_cleaned';
datasetName = 'tweet_sentiment_extraction';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

file = fullfile(datasetDir, [datasetName '_' modelName '.parquet']);

mkdir(imageDir, modelName);
mkdir(tableDir, modelName);

%% Load Data

N = 1000;
[eps20, eps20Labels] = generateSample(file, N, 0.2);

%% e=0.2, a=0.7

data = eps20;
labels = eps20Labels;
alpha = 0.7;

Y = tsne(data, Distance="cosine");
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, modelName, "e02_tsne.png"),'png');

clear Y;

kModel = AutoSphereRbfKernel(data);
% kModel = AutoRbfKernel(data);
poc = kMRCD(kModel); 
solution = poc.runAlgorithm(data, alpha);

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
stats = evaluation(data, labels, alpha, solution, Estimators={'lof' 'iforest'});
saveas(fig, fullfile(imageDir, modelName, 'e02_pr_curve.png'),'png');
writetable(stats, fullfile(tableDir, modelName, "e02_comparison.csv"));

clear stats;

clear solution kModel alpha poc;
clear data labels;
%% Comparison

[eps0, eps0Labels] = generateSample(file, N, 0);
[eps10, eps10Labels] = generateSample(file, N, 0.1);
[eps30, eps30Labels] = generateSample(file, N, 0.3);

stats = [
        runComparison(eps10, eps10Labels, 0.1, 0.5); runComparison(eps10, eps10Labels, 0.1, 0.75); runComparison(eps10, eps10Labels, 0.1, 0.9); ...
        runComparison(eps20, eps20Labels, 0.2, 0.5); runComparison(eps20, eps20Labels, 0.2, 0.75); runComparison(eps20, eps20Labels, 0.2, 0.9); ...
        runComparison(eps30, eps30Labels, 0.3, 0.5); runComparison(eps30, eps30Labels, 0.3, 0.75); runComparison(eps30, eps30Labels, 0.3, 0.9)
        ];

writetable(stats, fullfile(tableDir, modelName, 'comparison.csv'));

%% Functions

function [embeddings,labels] = generateSample(filepath, sampleSize, contamination)
    arguments
        filepath (1,1) string {mustBeFile}
        sampleSize (1,1) double {mustBeInteger, mustBePositive}
        contamination (1,1) double {mustBeInRange(contamination,0,1)}
    end

    rawData = parquetread(filepath, SelectedVariableNames=["text", "label_text", "embedding"]);
    rawData.label_text = categorical(rawData.label_text);

    positiveIndices = find(rawData.label_text == "positive");
    negativeIndices = find(rawData.label_text == "negative");
    
    numNegativeSamples = round(contamination * sampleSize);
    numPositiveSamples = sampleSize - numNegativeSamples;
    
    negativeSampleIndices = datasample(negativeIndices, numNegativeSamples, "Replace", false);
    positiveSampleIndices = datasample(positiveIndices, numPositiveSamples, "Replace", false);
    data = rawData(sort(cat(1, positiveSampleIndices, negativeSampleIndices)),:);
    
    embeddings = cell2mat(cellfun(@transpose,data.embedding, UniformOutput=false));
    labels = renamecats(data.label_text, {'positive' 'negative'}, {'inlier' 'outlier'});
    data.label_text = removecats(data.label_text);
    labels = removecats(labels);

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
    
    kModel = AutoSphereRbfKernel(data);
    poc = kMRCD(kModel); 
    solution = poc.runAlgorithm(data, robustness);

    stats = evaluation(data, labels, robustness, solution, Estimators={'lof' 'iforest'});

    names = string(stats.Properties.RowNames);
    e = repmat(outlierContamination,size(names));
    a = repmat(robustness,size(names));
    stats = [table(e,a,names, VariableNames={'e' 'a' 'Name'}), stats];
    stats.Properties.RowNames = {};
end