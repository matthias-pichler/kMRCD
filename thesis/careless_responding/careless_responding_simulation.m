%% Careless Responding Simulation

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'careless_responding_simulation';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

mkdir(imageDir);
mkdir(tableDir);

%% Visualize

[unlabeledData, labels] = loadData(directory=datasetDir, iteration=1, distribution="uni");

Y = tsne(unlabeledData);
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "tsne.png"),'png');

clear Y;

%% Uniform

filepath = fullfile(tableDir, "simulation_uni_a07.csv");
stats = runSimulation(directory=datasetDir, distribution="uni", alpha=0.7, maxIterations=100, file=filepath);

fig = figure(5);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_uni_a07.png'),'png');

%% Middle

filepath = fullfile(tableDir, "simulation_mid_a07.csv");
stats = runSimulation(directory=datasetDir, distribution="mid", alpha=0.7, maxIterations=100, file=filepath);

fig = figure(6);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_mid_a07.png'),'png');

%% Pattern

filepath = fullfile(tableDir, "simulation_pattern_a07.csv");
stats = runSimulation(directory=datasetDir, distribution="pattern", alpha=0.7, maxIterations=100, file=filepath);

fig = figure(7);
boxchart(stats.name, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_pattern_a07.png'),'png');

%% Functions

function [unlabeledData, labels] = loadData(NameValueArgs)
    arguments
        NameValueArgs.directory (1,1) string
        NameValueArgs.distribution (1,1) string {mustBeMember(NameValueArgs.distribution, ["mid" "uni" "pattern"])}
        NameValueArgs.iteration (1,1) double {mustBeInteger, mustBePositive}
    end
    
    file = fullfile(NameValueArgs.directory, sprintf("dat_%s", NameValueArgs.distribution), sprintf("dat_%d_%s.csv", NameValueArgs.iteration, NameValueArgs.distribution));
    
    opts = detectImportOptions(file);
    opts = setvartype(opts, 'double');
    
    data = readtable(file, opts);
    data.Careless = categorical(data.Careless, [0 1], {'regular', 'careless'});
    
    unlabeledData = table2array(removevars(data,{'Careless'}));
    labels = renamecats(data.Careless, {'regular' 'careless'}, {'inlier' 'outlier'});
    
    perm = randperm(height(unlabeledData));
    unlabeledData = unlabeledData(perm, :);
    labels = labels(perm, :);
end

function stats = runSimulation(NameValueArgs)
    arguments
        NameValueArgs.directory (1,1) string
        NameValueArgs.distribution (1,1) string {mustBeMember(NameValueArgs.distribution, ["mid" "uni" "pattern"])}
        NameValueArgs.alpha (1,1) double {mustBeInRange(NameValueArgs.alpha, 0.5, 1)}
        NameValueArgs.maxIterations (1,1) double {mustBeInteger, mustBePositive, mustBeLessThanOrEqual(NameValueArgs.maxIterations, 1000)} = 1000
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
        
        [data, labels] = loadData(directory=NameValueArgs.directory, distribution=NameValueArgs.distribution, iteration=i);
        
        % kModel = K1Kernel(data);
        kModel = StringSubsequenceKernel(lambda=0.05,maxSubsequence=5);
        poc = kMRCD(kModel);
        
        if isequal(class(kModel), 'StringSubsequenceKernel')
            encodedData = join(string(data), "");
            solution = poc.runAlgorithm(encodedData, alpha);
        else
            solution = poc.runAlgorithm(data, alpha);
        end
        
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