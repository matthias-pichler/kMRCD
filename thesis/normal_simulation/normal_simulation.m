%% Discretized Normal Distribution

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'normal_discretized';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

mkdir(imageDir);
mkdir(tableDir);

%% Visualize

[unlabeledData, labels] = loadData(directory=datasetDir, iteration=1, contamination=0.2, dimensions=30, categories=5);
[~, scores] = pca(unlabeledData);

Y = tsne(unlabeledData);
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "e02_tsne.png"),'png');

clear Y;

%% Sample

alpha = 0.7;

kModel = K1Kernel(unlabeledData);
% kModel = AutoRbfKernel(scores);

poc = kMRCD(kModel); 
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

clear stats cm grouphat;
clear solution kModel alpha poc;
clear data labels;

%% Run a = 0.7, e = 0.2

set(0,'DefaultFigureVisible','off');

filepath = fullfile(tableDir, "simulation_a07_e02.csv");
stats = runSimulation(iterations=100, alpha=0.7, file=filepath, dataDirectory=datasetDir, contamination=0.2, dimensions=30, categories=5);

fig = figure(5);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a07_e02.png'),'png');

%% Run a = 0.5, e = 0.2

filepath = fullfile(tableDir, "simulation_a05_e02.csv");
stats = runSimulation(iterations=100, alpha=0.5, file=filepath, dataDirectory=datasetDir, contamination=0.2, dimensions=30, categories=5);

fig = figure(6);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a05_e02.png'),'png');

%% Run a = 0.5, e = 0.3

filepath = fullfile(tableDir, "simulation_a05_e03.csv");
stats = runSimulation(iterations=100, alpha=0.5, file=filepath, dataDirectory=datasetDir, contamination=0.3, dimensions=30, categories=5);

fig = figure(7);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a05_e03.png'),'png');

%% Functions

function [data, labels] = loadData(NameValueArgs)
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

function stats = runSimulation(NameValueArgs)
    arguments
        NameValueArgs.iterations (1,1) double {mustBeInteger, mustBePositive} = 100
        NameValueArgs.alpha (1,1) double {mustBeInRange(NameValueArgs.alpha, 0.5, 1)}
        NameValueArgs.dataDirectory (1,1) string {mustBeFolder}
        NameValueArgs.contamination (1,1) double {mustBeInRange(NameValueArgs.contamination, 0, 0.5)}
        NameValueArgs.dimensions (1,1) double {mustBeInteger, mustBePositive}
        NameValueArgs.categories (1,1) double {mustBeInteger, mustBePositive}
        NameValueArgs.file (1,1) string
    end

    alpha = NameValueArgs.alpha;
    iter = NameValueArgs.iterations;
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

        [data, labels] = loadData(directory=NameValueArgs.dataDirectory, iteration=i, ...
            contamination=NameValueArgs.contamination, dimensions=NameValueArgs.dimensions, categories=NameValueArgs.categories);
        
        kModel = K1Kernel(data);
        
        poc = kMRCD(kModel); 
        solution = poc.runAlgorithm(data, alpha);
    
        e = evaluation(data, labels, alpha, solution, CategoricalPredictors="all");
    
        results = vertcat(e('kMRCD',:), e('lof',:), e('iforest', :), e('robustcov',:));
        results.name = ["kMRCD";"lof";"iforest";"robustcov"];
        results.iteration = repmat(i, 4, 1);
        
        writetable(results, file, WriteMode="append");
    end
    
    opts = detectImportOptions(file);
    opts = setvartype(opts, 'double');
    opts = setvartype(opts,'name', 'categorical');
    stats = readtable(file, opts);

    [path, filename, ext] = fileparts(file);
    mrcdFile = fullfile(path, sprintf("%s_mrcd%s", filename, ext));
    opts = detectImportOptions(mrcdFile);
    opts = setvartype(opts, 'double');
    opts = setvartype(opts,'name', 'categorical');
    mrcdStats = readtable(mrcdFile, opts);

    stats = vertcat(stats, mrcdStats);
end