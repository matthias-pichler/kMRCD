%% Discretized Normal Distribution Kernels

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'normal_discretized_kernels';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);

mkdir(imageDir);
mkdir(tableDir);

%% Run

set(0,'DefaultFigureVisible','off');

file = fullfile(tableDir, "simulation_a07_e02.csv");

alpha = 0.7;

iter = 100;
start = 1;

if isfile(file)
    opts = detectImportOptions(file);
    opts = setvartype(opts, 'double');
    opts = setvartype(opts,'kernel', 'categorical');
    results = readtable(file, opts);
    start = max(results.iteration) + 1;
end

for i = start:iter
    fprintf("Iteration: %d\n", i);

    [data, labels] = generateData(size=1000, contamination=0.2, dimensions=30, categories=5);
    
    s = struct();
    %%% Linear
    kModel = LinKernel();
    solution = kMRCD(kModel).runAlgorithm(data, alpha);
    
    s(1).kernel = "Linear";
    s(1).solution = solution;
    
    %%% RBF
    kModel = AutoRbfKernel(data);
    solution = kMRCD(kModel).runAlgorithm(data, alpha);
    
    s(2).kernel = "RBF";
    s(2).solution = solution;
    
    %%% Dirac
    kModel = DiracKernel();
    solution = kMRCD(kModel).runAlgorithm(data, alpha);
    s(3).kernel = "Dirac";
    s(3).solution = solution;
    
    %%% k1
    kModel = K1Kernel(data);
    solution = kMRCD(kModel).runAlgorithm(data, alpha);
    s(4).kernel = "k1";
    s(4).solution = solution;
    
    %%% m3
    kModel = M3Kernel(data);
    solution = kMRCD(kModel).runAlgorithm(data, alpha);
    s(5).kernel = "m3";
    s(5).solution = solution;
    
    %%% Aitchison-Aitken
    kModel = AitchisonAitkenKernel(data);
    solution = kMRCD(kModel).runAlgorithm(data, alpha);
    s(6).kernel = "Aitchison-Aitken";
    s(6).solution = solution;
    
    %%% Li-Racin
    kModel = LiRacinKernel(data);
    solution = kMRCD(kModel).runAlgorithm(data, alpha);
    s(7).kernel = "Li-Racin";
    s(7).solution = solution;

    %%% Wang-Ryzin
    kModel = WangRyzinKernel(data, lambda=repmat(0.01, 1, width(data)));
    solution = kMRCD(kModel).runAlgorithm(data, alpha);
    s(8).kernel = "Wang-Ryzin";
    s(8).solution = solution;
    
    %%% Ordered Aitchison-Aitken
    kModel = OrderedAitchisonAitkenKernel(data);
    solution = kMRCD(kModel).runAlgorithm(data, alpha);
    s(9).kernel = "Ordered Aitchison-Aitken";
    s(9).solution = solution;
    
    %%% Ordered Li-Racin
    kModel = OrderedLiRacinKernel(data, lambda=repmat(0.01, 1, width(data)));
    solution = kMRCD(kModel).runAlgorithm(data, alpha);
    s(10).kernel = "Ordered Li-Racin";
    s(10).solution = solution;

    colors = jet(numel(s));

    %%% Summary
    for j = 1:numel(s)
        grouphat = categorical(repmat("inlier", size(labels)), categories(labels));
        grouphat(s(j).solution.flaggedOutlierIndices) = "outlier";
        cm = confusionmat(labels,grouphat, Order={'outlier' 'inlier'});
        stats = confusionstats(cm);
    
        s(j).accuracy = stats.accuracy;
        s(j).precision = stats.precision;
        s(j).sensitivity = stats.sensitivity;
        s(j).specificity = stats.specificity;
        s(j).f1Score = stats.f1Score;

        scores = s(j).solution.rd;
        auc = prcurve(labels,scores,'outlier',DisplayName=s(j).kernel, Color=colors(j,:));
        s(j).aucpr = auc;
    end

    results = struct2table(rmfield(s, "solution"));
    results.iteration = repmat(i, numel(s), 1);
    writetable(results, file, WriteMode="append");
end

opts = detectImportOptions(file);
opts = setvartype(opts, 'double');
opts = setvartype(opts,'name', 'categorical');
stats = readtable(file, opts);

fig = figure(1);
boxchart(stats.kernel, stats.aucpr, MarkerStyle="none");
saveas(fig, fullfile(imageDir, 'boxplot_a07_e02.png'),'png');

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