%% Normal Distribution Data

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

datasetName = 'bands';

datasetDir = fullfile(projectDir, 'datasets', datasetName);

mkdir(datasetDir);

%% Run
for eps = [0, 0.2]
    for p = [10, 50, 100, 500, 750, 1000]
        [data, labels] = generateData(size=1000, contamination=eps, dimensions=p);
    
        res = horzcat(array2table(data), array2table(labels));
        
        file = fullfile(datasetDir, sprintf("data_d%d_e0%.0f.csv", p, eps * 10));
        writetable(res, file);
    end
end

%% Functions

function [data, labels] = generateData(NameValueArgs)
    arguments
        NameValueArgs.size (1,1) double {mustBeInteger, mustBePositive}
        NameValueArgs.contamination (1,1) double {mustBeInRange(NameValueArgs.contamination, 0, 0.5)}
        NameValueArgs.dimensions (1,1) double {mustBeInteger, mustBePositive}
    end

    contamination = NameValueArgs.contamination;
    dimensions = NameValueArgs.dimensions;
    N = NameValueArgs.size;
    
    ndm = NewDataModel(ALYZCorrelationType(), ClusterContamination());
    [data, ~, ~,idxOutliers] = ndm.generateDataset(N, dimensions, contamination, 20);        
    
    labels = categorical(repmat("inlier", [N 1]), {'inlier' 'outlier'});
    labels(idxOutliers) = "outlier";
end