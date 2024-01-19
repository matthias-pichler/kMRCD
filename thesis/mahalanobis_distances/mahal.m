%% Discretized Normal Distribution

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

imageDir = fullfile(projectDir, 'images', 'mahalanobis_distances');
tableDir = fullfile(projectDir, 'tables', 'mahalanobis_distances');

mkdir(imageDir);
mkdir(tableDir);

normalDatasetDir = fullfile(projectDir, 'datasets', 'normal_discretized');
shakespearTrumpDatasetDir = fullfile(projectDir, 'datasets', 'shakespear_trump');

%% Run (Normal)

mkdir(fullfile(imageDir, 'normal_discretized'))

alpha = 0.7;
[unlabeledData, labels] = loadNormalData(directory=normalDatasetDir, iteration=1, contamination=0.2, dimensions=30, categories=5);
[~, scores] = pca(unlabeledData);

kModel = K1Kernel(unlabeledData);

poc = kMRCD(kModel); 
solution = poc.runAlgorithm(unlabeledData, alpha);

% Mahalanobis Distances
fig = figure(1);
mahalchart(labels, solution.rd, solution.cutoff);
saveas(fig, fullfile(imageDir, 'normal_discretized', 'mahalanobis_distances.png'),'png');


kModel = AutoRbfKernel(scores);
poc = kMRCD(kModel); 
solution = poc.runAlgorithm(unlabeledData, alpha);

% Mahalanobis Distances
fig = figure(2);
mahalchart(labels, solution.rd, solution.cutoff);
saveas(fig, fullfile(imageDir, 'normal_discretized', 'mahalanobis_distances_pca.png'),'png');

%% Run Shakespear

mkdir(fullfile(imageDir, 'shakespear_trump'))

[embeddings, labels] = generateShakespearTrumpSample(directory=shakespearTrumpDatasetDir, sampleSize=1000, contamination=0.2, model="all-mpnet-base-v2");
[~, scores] = pca(embeddings);

kModel = AutoSphereRbfKernel(embeddings);

poc = kMRCD(kModel); 
solution = poc.runAlgorithm(embeddings, alpha);

% Mahalanobis Distances
fig = figure(3);
mahalchart(labels, solution.rd, solution.cutoff);
saveas(fig, fullfile(imageDir, 'shakespear_trump', 'mahalanobis_distances.png'),'png');


kModel = AutoRbfKernel(scores);
poc = kMRCD(kModel); 
solution = poc.runAlgorithm(scores, alpha);

% Mahalanobis Distances
fig = figure(4);
mahalchart(labels, solution.rd, solution.cutoff);
saveas(fig, fullfile(imageDir, 'shakespear_trump', 'mahalanobis_distances_pca.png'),'png');

%% Functions

function [data, labels] = loadNormalData(NameValueArgs)
    arguments
        NameValueArgs.directory (1,1) string {mustBeFolder}
        NameValueArgs.iteration (1,1) double {mustBeInteger, mustBePositive}
        NameValueArgs.contamination (1,1) double {mustBeInRange(NameValueArgs.contamination, 0, 0.5)}
        NameValueArgs.dimensions (1,1) double {mustBeInteger, mustBePositive}
        NameValueArgs.categories (1,1) double {mustBeInteger, mustBePositive}
    end

    dataDir = fullfile(NameValueArgs.directory, sprintf("data_c%d_d%d_e0%.0f", NameValueArgs.categories, NameValueArgs.dimensions, NameValueArgs.contamination * 10));
    dataFile = fullfile(dataDir, sprintf("data_%d.csv", NameValueArgs.iteration));

    opts = detectImportOptions(dataFile);
    opts = setvartype(opts, 'double');
    opts = setvartype(opts,'labels', 'categorical');
    data = readtable(dataFile, opts);

    labels = data.labels;
    data = table2array(removevars(data, {'labels'}));
end

function [embeddings,labels] = generateShakespearTrumpSample(NameValueArgs)
    arguments
        NameValueArgs.directory (1,1) string {mustBeFolder}
        NameValueArgs.model (1,1) string
        NameValueArgs.sampleSize (1,1) double {mustBeInteger, mustBePositive}
        NameValueArgs.contamination (1,1) double {mustBeInRange(NameValueArgs.contamination, 0, 0.5)}
    end

    filepath = fullfile(NameValueArgs.directory, sprintf("shakespear_trump_%s.parquet", NameValueArgs.model));

    rawData = parquetread(filepath, SelectedVariableNames=["text", "author", "embedding"]);
    rawData.author = categorical(rawData.author);

    shakespearIndices = find(rawData.author == "shakespear");
    trumpIndices = find(rawData.author == "trump");
    
    numShakespearSamples = round(NameValueArgs.contamination * NameValueArgs.sampleSize);
    numTrumpSamples = NameValueArgs.sampleSize - numShakespearSamples;
    
    shakespearSampleIndices = datasample(shakespearIndices, numShakespearSamples, Replace=false);
    trumpSampleIndices = datasample(trumpIndices, numTrumpSamples, Replace=false);
    data = rawData(sort(cat(1, shakespearSampleIndices, trumpSampleIndices)),:);

    embeddings = cell2mat(cellfun(@transpose,data.embedding, UniformOutput=false));
    labels = renamecats(data.author, {'trump' 'shakespear'}, {'inlier' 'outlier'});

    perm = randperm(height(embeddings));
    embeddings = embeddings(perm, :);
    labels = labels(perm, :);
end
