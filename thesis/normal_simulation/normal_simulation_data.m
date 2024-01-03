%% Discretized Normal Distribution Kernels

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'normal_discretized';

datasetDir = fullfile(projectDir, 'datasets', datasetName);

mkdir(datasetDir);

%% Run

iter = 100;
start = 1;

p = 30;
c = 5;
eps = 0.2;

for i = start:iter
    fprintf("Iteration: %d\n", i);

    [data, labels] = generateData(size=500, contamination=eps, dimensions=p, categories=c);
    
    res = horzcat(array2table(data), array2table(labels));

    file = fullfile(datasetDir, sprintf("data_c%d_d%d_e0%.0f_%d.csv", c, p, eps * 10, i));
    writetable(res, file);
end

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