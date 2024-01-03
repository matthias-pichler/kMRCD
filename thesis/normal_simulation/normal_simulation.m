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

mkdir(imageDir);
mkdir(tableDir);

%% Visualize

[unlabeledData, labels] = generateData(size=1000, contamination=0.2, dimensions=30, categories=5);

Y = tsne(unlabeledData);
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "e02_tsne.png"),'png');

clear Y;

%% Sample

alpha = 0.7;

kModel = K1Kernel(unlabeledData);
% kModel = AutoRbfKernel(unlabeledData);

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
stats = runSimulation(iter=100, alpha=0.7, file=filepath, data=@()generateData(size=1000, contamination=0.2, dimensions=30, categories=5));

fig = figure(5);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a07_e02.png'),'png');

%% Run a = 0.5, e = 0.2

filepath = fullfile(tableDir, "simulation_a05_e02.csv");
stats = runSimulation(iter=100, alpha=0.5, file=filepath, data=@()generateData(size=1000, contamination=0.2, dimensions=30, categories=5));

fig = figure(6);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a05_e02.png'),'png');

%% Run a = 0.5, e = 0.3

filepath = fullfile(tableDir, "simulation_a05_e03.csv");
stats = runSimulation(iter=100, alpha=0.5, file=filepath, data=@()generateData(size=1000, contamination=0.3, dimensions=30, categories=5));

fig = figure(7);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a05_e03.png'),'png');

%% Functions

function [data, labels] = generateData(NameValueArgs)
    arguments
        NameValueArgs.size (1,1) double {mustBeInteger, mustBePositive}
        NameValueArgs.contamination (1,1) double {mustBeInRange(NameValueArgs.contamination, 0, 0.5)}
        NameValueArgs.dimensions (1,1) double {mustBeInteger, mustBePositive}
        NameValueArgs.categories (1,1) double {mustBeInteger, mustBePositive}
    end

    contamination = NameValueArgs.contamination;
    numCategories = NameValueArgs.categories;
    dimensions = NameValueArgs.dimensions;
    N = NameValueArgs.size;
    
    ndm = NewDataModel(ALYZCorrelationType(), ClusterContamination());
    [x, ~, ~,idxOutliers] = ndm.generateDataset(N, dimensions, contamination, 20);        
    
    data = cell2mat(cellfun(@(X)discretize(X, numCategories), num2cell(x, 1), UniformOutput=false));
    
    labels = categorical(repmat("inlier", [N 1]), {'inlier' 'outlier'});
    labels(idxOutliers) = "outlier";
end

function stats = runSimulation(NameValueArgs)
    arguments
        NameValueArgs.iter (1,1) double {mustBeInteger, mustBePositive} = 100
        NameValueArgs.alpha (1,1) double {mustBeInRange(NameValueArgs.alpha, 0.5, 1)}
        NameValueArgs.data
        NameValueArgs.file (1,1) string
    end

    alpha = NameValueArgs.alpha;
    iter = NameValueArgs.iter;
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

        if(isa(NameValueArgs.data, 'function_handle'))
            [data, labels] = NameValueArgs.data();
        end
        
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
end