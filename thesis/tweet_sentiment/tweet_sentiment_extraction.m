%% Tweet Sentiment
% Compute the case study of shakespear vs trump

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

modelName = 'all-mpnet-base-v2';
% modelName = 'bge-large-en-v1.5';
% modelName = 'bge-small-en-v1.5';
% modelName = 'all-MiniLM-L6-v2';

% datasetName = 'tweet_sentiment_extraction_cleaned';
datasetName = 'tweet_sentiment_extraction';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

datasetFile = fullfile(datasetDir, [datasetName '_' modelName '.parquet']);

mkdir(imageDir, modelName);
mkdir(tableDir, modelName);

imageDir = fullfile(imageDir, modelName);
tableDir = fullfile(tableDir, modelName);

%% Visualize

[embeddings, labels] = generateSample(datasetFile, 1000, 0.2);

Y = tsne(embeddings, Distance="cosine");
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "tsne.png"),'png');

clear Y;

%% Sample

alpha = 0.7;
N = 1000;

kModel = AutoSphereRbfKernel(embeddings);
% kModel = AutoRbfKernel(embeddings);

poc = kMRCD(kModel); 
solution = poc.runAlgorithm(embeddings, alpha);

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
stats = evaluation(embeddings, labels, alpha, solution, Estimators={'lof' 'iforest'});
saveas(fig, fullfile(imageDir, 'pr_curve.png'),'png');

clear stats;
clear solution kModel alpha poc;
clear embeddings labels;

set(0,'DefaultFigureVisible','off');
%% Run a = 0.5, e = 0.1

filepath = fullfile(tableDir, "simulation_a05_e01.csv");
stats = runComparison(maxIterations=100, alpha=0.5, contamination=0.1, dataset=datasetFile, file=filepath);

fig = figure(5);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a05_e01.png'),'png');

%% Run a = 0.75, e = 0.1

filepath = fullfile(tableDir, "simulation_a075_e01.csv");
stats = runComparison(maxIterations=100, alpha=0.75, contamination=0.1, dataset=datasetFile, file=filepath);

fig = figure(6);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a075_e01.png'),'png');

%% Run a = 0.9, e = 0.1

filepath = fullfile(tableDir, "simulation_a09_e01.csv");
stats = runComparison(maxIterations=100, alpha=0.9, contamination=0.1, dataset=datasetFile, file=filepath);

fig = figure(7);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a09_e01.png'),'png');

%% Run a = 0.5, e = 0.2

filepath = fullfile(tableDir, "simulation_a05_e02.csv");
stats = runComparison(maxIterations=100, alpha=0.5, contamination=0.2, dataset=datasetFile, file=filepath);

fig = figure(8);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a05_e02.png'),'png');

%% Run a = 0.75, e = 0.2

filepath = fullfile(tableDir, "simulation_a075_e02.csv");
stats = runComparison(maxIterations=100, alpha=0.75, contamination=0.2, dataset=datasetFile, file=filepath);

fig = figure(9);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a075_e02.png'),'png');

%% Run a = 0.5, e = 0.3

filepath = fullfile(tableDir, "simulation_a05_e03.csv");
stats = runComparison(maxIterations=100, alpha=0.5, contamination=0.3, dataset=datasetFile, file=filepath);

fig = figure(10);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a05_e03.png'),'png');

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

function stats = runComparison(NameValueArgs)
    arguments
        NameValueArgs.dataset (1,1) string {mustBeFile}
        NameValueArgs.maxIterations (1,1) double {mustBeInteger, mustBePositive} = 100
        NameValueArgs.alpha (1,1) double {mustBeInRange(NameValueArgs.alpha, 0.5, 1)}
        NameValueArgs.contamination (1,1) double {mustBeInRange(NameValueArgs.contamination, 0, 0.5)}
        NameValueArgs.file (1,1) string
    end

    alpha = NameValueArgs.alpha;
    iter = NameValueArgs.maxIterations;
    file = NameValueArgs.file;

    start = 1;

    if isfile(file)
        opts = detectImportOptions(file);
        opts = setvartype(opts, 'double');
        opts = setvartype(opts,'name', 'categorical');
        results = readtable(file, opts);
        start = max(results.iteration) + 1;
    end

    for i = start:iter
        fprintf("Iteration: %d\n", i);

        [data, labels] = generateSample(NameValueArgs.dataset, 1000, NameValueArgs.contamination);
        
        kModel = AutoSphereRbfKernel(data);
        
        poc = kMRCD(kModel); 
        solution = poc.runAlgorithm(data, alpha);
    
        e = evaluation(data, labels, alpha, solution, Estimators={'lof' 'iforest'});
    
        results = vertcat(e('kMRCD',:), e('lof',:), e('iforest', :));
        results.name = ["kMRCD";"lof";"iforest"];
        results.iteration = repmat(i, 3, 1);
        
        writetable(results, file, WriteMode="append");
    end
    
    opts = detectImportOptions(file);
    opts = setvartype(opts, 'double');
    opts = setvartype(opts,'name', 'categorical');
    stats = readtable(file, opts);
end