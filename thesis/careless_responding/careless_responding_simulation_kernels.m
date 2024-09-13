%% Careless Responding Simulation (Kernel comparison)

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'careless_responding_simulation';

imageDir = fullfile(projectDir, 'images', [datasetName '_kernels']);
tableDir = fullfile(projectDir, 'tables', [datasetName '_kernels']);
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

function encodings = diffEncode(data)
    arguments
        data (:,:) double
    end
    
    differences = table2array(diff(data, 1, 2));
    firstColumn = data.(1);

    encodings = horzcat(firstColumn, differences);
    encodings = array2table(encodings, VariableNames=data.Properties.VariableNames);
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
    
    function res = run(kModel, unlabeledData, name)
        poc = kMRCD(kModel);

        if isequal(class(kModel), 'StringSubsequenceKernel')
            stringEncodedData = join(string(unlabeledData), "");
            solution = poc.runAlgorithm(stringEncodedData, alpha);
        else
            solution = poc.runAlgorithm(unlabeledData, alpha);
        end

        grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
        grouphat(solution.flaggedOutlierIndices) = "outlier";
        stats = confusionstats(confusionmat(labels,grouphat, Order={'outlier' 'inlier'}));
        
        res = struct2table(stats);
        res.name = name;
    end

    for i = start:iter
        fprintf("Iteration: %d\n", i);
        
        [unlabeledData, labels] = loadData(directory=NameValueArgs.directory, distribution=NameValueArgs.distribution, iteration=i);

        %% Linear
        kModel = LinKernel();
        results = run(kModel, unlabeledData, "Linear");

        %% RBF
        kModel = AutoRbfKernel(unlabeledData);
        results = vertcat(results, run(kModel, unlabeledData, "RBF"));

        %% Dirac
        kModel = DiracKernel();
        results = vertcat(results, run(kModel, unlabeledData, "Dirac"));

        %% k1
        kModel = K1Kernel(unlabeledData);
        results = vertcat(results, run(kModel, unlabeledData, "k1"));

        %% m3
        kModel = M3Kernel(unlabeledData);
        results = vertcat(results, run(kModel, unlabeledData, "m3"));

        %% Aitchison-Aitken
        kModel = AitchisonAitkenKernel(unlabeledData);
        results = vertcat(results, run(kModel, unlabeledData, "Aitchison-Aitken"));

        %% Li-Racin
        kModel = LiRacinKernel(unlabeledData);
        results = vertcat(results, run(kModel, unlabeledData, "Li-Racin"));

        %% String-Subsequence
        kModel = StringSubsequenceKernel(maxSubsequence=15, lambda=0.6);
        results = vertcat(results, run(kModel, unlabeledData, "SSK"));

        %% Ordered Aitchison-Aitken
        kModel = OrderedAitchisonAitkenKernel(unlabeledData);
        results = vertcat(results, run(kModel, unlabeledData, "Ordered Aitchison-Aitken"));

        %% Ordered Li-Racin
        kModel = OrderedLiRacinKernel(unlabeledData, lambda=repmat(0.01, 1, width(unlabeledData)));
        results = vertcat(results, run(kModel, unlabeledData, "Ordered Li-Racin"));

        %% Wang-Ryzin
        kModel = WangRyzinKernel(unlabeledData, lambda=repmat(0.01, 1, width(unlabeledData)));
        results = vertcat(results, run(kModel, unlabeledData, "Wang-Ryzin"));

        results.iteration = repmat(i, height(results), 1);
        writetable(results, file, WriteMode="append");
    end
    
    opts = detectImportOptions(file);
    opts = setvartype(opts, 'double');
    opts = setvartype(opts,'name', 'categorical');
    stats = readtable(file, opts);
end