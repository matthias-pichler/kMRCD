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

[unlabeledData, labels] = loadData(directory=datasetDir, iteration=1, type="uni");

Y = tsne(unlabeledData);
fig = figure(1);
gscatter(Y(:,1), Y(:,2), labels);
title("t-SNE Embeddings");
saveas(fig,fullfile(imageDir, "tsne.png"),'png');

clear Y;

%% Sample

alpha = 0.7;

kModel = K1Kernel(unlabeledData);

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
stats = evaluation(unlabeledData, labels, alpha, solution);
saveas(fig, fullfile(imageDir, 'pr_curve.png'),'png');

clear stats cm grouphat;
clear solution kModel alpha poc;
clear data labels;

%% Uniform

stats = runSimulation(directory=datasetDir, type="uni", alpha=0.7);

writetable(stats, fullfile(tableDir, "comparison_uni_a07.csv"));

%% Middle

stats = runSimulation(directory=datasetDir, type="mid", alpha=0.7);

writetable(stats, fullfile(tableDir, "comparison_mid_a07.csv"));

%% Pattern

stats = runSimulation(directory=datasetDir, type="pattern", alpha=0.7);

writetable(stats, fullfile(tableDir, "comparison_pattern_a07.csv"));

%% Functions

function [unlabeledData, labels] = loadData(NameValueArgs)
    arguments
        NameValueArgs.directory (1,1) string
        NameValueArgs.type (1,1) string {mustBeMember(NameValueArgs.type, ["mid" "uni" "pattern"])}
        NameValueArgs.iteration (1,1) double {mustBeInteger, mustBePositive}
    end

    file = fullfile(NameValueArgs.directory, sprintf("dat_%s", NameValueArgs.type), sprintf("dat_%d_%s.csv", NameValueArgs.iteration, NameValueArgs.type));
    
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
        NameValueArgs.type (1,1) string {mustBeMember(NameValueArgs.type, ["mid" "uni" "pattern"])}
        NameValueArgs.alpha (1,1) double {mustBeInRange(NameValueArgs.alpha, 0.5, 1)}
        NameValueArgs.maxIterations (1,1) double {mustBeInteger, mustBePositive, mustBeLessThanOrEqual(NameValueArgs.maxIterations, 1000)} = 1000
    end

    alpha = NameValueArgs.alpha;
    
    kMRCDStats = table(Size=[1000, 6], ...
        VariableNames={'accuracy' 'precision' 'sensitivity' 'specificity' 'f1Score' 'aucpr'}, ...
        VariableTypes=repmat("double", 1, 6));
    lofStats = kMRCDStats;
    iforestStats = kMRCDStats;
    robustcovStats = kMRCDStats;
    
    for i = 1:1000
        fprintf("Iteration: %d\n", i);

        [data, labels] = loadData(directory=NameValueArgs.directory, type=NameValueArgs.type, iteration=i);
        
        kModel = K1Kernel(data);
        
        poc = kMRCD(kModel); 
        solution = poc.runAlgorithm(data, alpha);
    
        e = evaluation(data, labels, alpha, solution, CategoricalPredictors="all");
    
        kMRCDStats(i,:) = e('kMRCD',:);
        lofStats(i,:) = e('lof', :);
        iforestStats(i,:) = e('iforest', :);
        robustcovStats(i,:) = e('robustcov',:);
    end
    
    stats = vertcat(harmmean(kMRCDStats), harmmean(lofStats), harmmean(iforestStats),harmmean(robustcovStats));
    stats = horzcat(table(["kMRCD";"lof";"iforest";"robustcov"], VariableNames={'name'}),stats);
    stats.Properties.RowNames = stats.name;
end