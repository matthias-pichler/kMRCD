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
mkdir(tableDir);

%% Run

alpha = 0.7;

for eps = [0, 0.2]
    for p = [10, 50, 100, 500, 750, 1000]
        [unlabeledData, labels] = loadData(directory=datasetDir, contamination=eps, dimensions=p);
    
        kModel = AutoRbfKernel(unlabeledData);
        
        poc = kMRCD(kModel); 
        solution = poc.runAlgorithm(unlabeledData, alpha);
        
        % Mahalanobis Distances
        fig = figure();
        mahalchart(labels, solution.rd, solution.cutoff);
        saveas(fig, fullfile(imageDir, sprintf("mahalanobis_distances_d%d_e0%.0f.png", p, eps*10)),'png');
    end
end


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