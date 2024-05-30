%% Normal Distribution

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'bands';

imageDir = fullfile(projectDir, 'images', datasetName);
tableDir = fullfile(projectDir, 'tables', datasetName);
datasetDir = fullfile(projectDir, 'datasets', datasetName);

mkdir(imageDir);
mkdir(fullfile(imageDir, "AutoRbfKernel"));
mkdir(fullfile(imageDir, "ScaledAutoRbfKernel"));
mkdir(fullfile(imageDir, "LinKernel"));
mkdir(fullfile(imageDir, "DetMCD"));
mkdir(tableDir);

%% Run

alpha = 0.7;
eps = 0;
dims = [5, 10, 25, 50, 75, 100:50:1000];
gaps = table(size=[length(dims), 5], VariableTypes=repmat("double", 1, 5), VariableNames={'Dimensions', 'AutoRbfKernel', 'ScaledAutoRbfKernel', 'LinKernel', 'DetMCD'});
gaps(:,:) = array2table(nan(size(gaps)));
gaps.Dimensions = dims';

%% AutoRbf

for i = 1:length(dims)
    p = dims(i);
    [unlabeledData, labels] = loadData(directory=datasetDir, contamination=eps, dimensions=p);
    
    kModel = AutoRbfKernel(unlabeledData);
    solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
    
    % Mahalanobis Distances
    fig = figure();
    hSubset = categorical(repmat("not in H", [height(unlabeledData) 1]), {'in H' 'not in H'});
    hSubset(solution.hsubsetIndices) = "in H";
    mahalchart(hSubset, solution.rd, solution.cutoff);
    saveas(fig, fullfile(imageDir, "AutoRbfKernel", sprintf("mahalanobis_distances_d%d_e0%.0f.png", p, eps*10)),'png');
    
    gaps.AutoRbfKernel(i) = gap(hSubset, solution.rd);
end

%% ScaledAutoRbf

for i = 1:length(dims)
    p = dims(i);
    [unlabeledData, labels] = loadData(directory=datasetDir, contamination=eps, dimensions=p);
    
    kModel = ScaledAutoRbfKernel(unlabeledData);
    solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
    
    % Mahalanobis Distances
    fig = figure();
    hSubset = categorical(repmat("not in H", [height(unlabeledData) 1]), {'in H' 'not in H'});
    hSubset(solution.hsubsetIndices) = "in H";
    mahalchart(hSubset, solution.rd, solution.cutoff);
    saveas(fig, fullfile(imageDir, "ScaledAutoRbfKernel", sprintf("mahalanobis_distances_d%d_e0%.0f.png", p, eps*10)),'png');
    
    gaps.ScaledAutoRbfKernel(i) = gap(hSubset, solution.rd);
end

%% Linear

for i = 1:length(dims)
    p = dims(i);
    [unlabeledData, labels] = loadData(directory=datasetDir, contamination=eps, dimensions=p);
    
    rbfModel = LinKernel();
    solution = kMRCD(rbfModel).runAlgorithm(unlabeledData, alpha);
    
    % Mahalanobis Distances
    fig = figure();
    hSubset = categorical(repmat("not in H", [height(unlabeledData) 1]), {'in H' 'not in H'});
    hSubset(solution.hsubsetIndices) = "in H";
    mahalchart(hSubset, solution.rd, solution.cutoff);
    saveas(fig, fullfile(imageDir, "LinKernel", sprintf("mahalanobis_distances_d%d_e0%.0f.png", p, eps*10)),'png');
    
    gaps.LinKernel(i) = gap(hSubset, solution.rd);
end

%% MCD

for i = 1:length(dims)
    p = dims(i);
    [unlabeledData, labels] = loadData(directory=datasetDir, contamination=eps, dimensions=p);
    
    solution = LIBRA.DetMCD(unlabeledData, 'alpha', alpha, 'plots', 0);
    
    % Mahalanobis Distances
    fig = figure();
    hSubset = categorical(repmat("not in H", [height(unlabeledData) 1]), {'in H' 'not in H'});
    hSubset(solution.Hsubsets.Hopt) = "in H";
    mahalchart(hSubset, solution.rd, solution.cutoff.rd);
    saveas(fig, fullfile(imageDir, "DetMCD", sprintf("mahalanobis_distances_d%d_e0%.0f.png", p, eps*10)),'png');
    
    gaps.DetMCD(i) = gap(hSubset, solution.rd);
end

%% Gaps

fig = figure();
plot(gaps,"Dimensions",["AutoRbfKernel", "ScaledAutoRbfKernel", "LinKernel", "DetMCD"]);
ylabel("Gap size");
legend;
saveas(fig, fullfile(imageDir, sprintf("gaps_e0%.0f.png", eps*10)),'png');

%% Functions

function [data, labels] = loadData(NameValueArgs)
    arguments
        NameValueArgs.directory (1,1) string {mustBeFolder}
        NameValueArgs.contamination (1,1) double {mustBeInRange(NameValueArgs.contamination, 0, 0.5)}
        NameValueArgs.dimensions (1,1) double {mustBeInteger, mustBePositive}
    end
    
    dataFile = fullfile(NameValueArgs.directory, sprintf("data_d%d_e0%.0f.csv", NameValueArgs.dimensions, NameValueArgs.contamination * 10));
    
    opts = detectImportOptions(dataFile);
    opts = setvartype(opts, 'double');
    opts = setvartype(opts,'labels', 'categorical');
    data = readtable(dataFile, opts);
    
    labels = data.labels;
    data = table2array(removevars(data, {'labels'}));
end

function g = gap(labels, distances)
    arguments
        labels (:,1) categorical
        distances (:,1) double
    end
    
    assert(isequal(size(labels), size(distances)));
    
    groups = splitapply(@(x){x}, distances, double(labels));

    [X,Y] = groups{:};
    
    g = min(pdist2(X,Y, "euclidean", Smallest=1));
end