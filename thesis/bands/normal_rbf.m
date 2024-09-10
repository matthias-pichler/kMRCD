%% Normal Distribution

% Setup
clc;
clear all;
close all;
rng(1634256, "twister");

fileDir = fileparts(which(mfilename));
projectDir = fileparts(fileparts(fileDir));

imageDir = fullfile(projectDir, 'images', 'normal_rbf');
tableDir = fullfile(projectDir, 'tables', 'normal_rbf');
datasetDir = fullfile(projectDir, 'datasets', 'bands');

mkdir(imageDir);
mkdir(fullfile(imageDir, "AutoRbfKernel (Median)"));
mkdir(fullfile(imageDir, "AutoRbfKernel (Scaled Median)"));
mkdir(fullfile(imageDir, "AutoRbfKernel (Mean)"));
mkdir(fullfile(imageDir, "AutoRbfKernel (Modified Mean)"));
mkdir(fullfile(imageDir, "RbfKernel (p^1/2)"));
mkdir(tableDir);

%% Run

alpha = 0.7;
eps = 0;
dims = [5, 10, 25, 50, 75, 100:50:1000];
gaps = table(size=[length(dims), 6], VariableTypes=repmat("double", 1, 6), VariableNames={'AutoRbfKernel (Median)', 'AutoRbfKernel (Scaled Median)', 'AutoRbfKernel (Mean)', 'AutoRbfKernel (Modified Mean)', 'RbfKernel (p^1/2)', 'Dimensions'});
gaps(:,:) = array2table(nan(size(gaps)));
gaps.Dimensions = dims';

%% AutoRbf (Median)
for i = 1:length(dims)
    p = dims(i);
    [unlabeledData, labels] = loadData(directory=datasetDir, contamination=eps, dimensions=p);

    kModel = AutoRbfKernel(unlabeledData, bandwidth="median");
    solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
    
    hSubset = categorical(repmat("not in H", [height(unlabeledData) 1]), {'in H' 'not in H'});
    hSubset(solution.hsubsetIndices) = "in H";
    
    % Mahalanobis Distances
    % fig = figure();
    % mahalchart(hSubset, solution.rd, solution.cutoff);
    % saveas(fig, fullfile(imageDir, "AutoRbfKernel (Median)", sprintf("mahalanobis_distances_d%d_e0%.0f.png", p, eps*10)),'png');
    
    gaps{i, 1} = gap(hSubset, solution.rd);
end

%% AutoRbf (Scaled Median)
for i = 1:length(dims)
    p = dims(i);
    [unlabeledData, labels] = loadData(directory=datasetDir, contamination=eps, dimensions=p);

    kModel = AutoRbfKernel(unlabeledData, bandwidth="scaledmedian");
    solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
    
    hSubset = categorical(repmat("not in H", [height(unlabeledData) 1]), {'in H' 'not in H'});
    hSubset(solution.hsubsetIndices) = "in H";
    
    % Mahalanobis Distances
    % fig = figure();
    % mahalchart(hSubset, solution.rd, solution.cutoff);
    % saveas(fig, fullfile(imageDir, "AutoRbfKernel (Scaled Median)", sprintf("mahalanobis_distances_d%d_e0%.0f.png", p, eps*10)),'png');
    
    gaps{i, 2} = gap(hSubset, solution.rd);
end

%% AutoRbf (Mean)
for i = 1:length(dims)
    p = dims(i);
    [unlabeledData, labels] = loadData(directory=datasetDir, contamination=eps, dimensions=p);

    kModel = AutoRbfKernel(unlabeledData, bandwidth="mean");
    solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
    
    hSubset = categorical(repmat("not in H", [height(unlabeledData) 1]), {'in H' 'not in H'});
    hSubset(solution.hsubsetIndices) = "in H";
    
    % Mahalanobis Distances
    % fig = figure();
    % mahalchart(hSubset, solution.rd, solution.cutoff);
    % saveas(fig, fullfile(imageDir, "AutoRbfKernel (Scaled Median)", sprintf("mahalanobis_distances_d%d_e0%.0f.png", p, eps*10)),'png');
    
    gaps{i, 3} = gap(hSubset, solution.rd);
end

%% AutoRbf (Modified Mean)
for i = 1:length(dims)
    p = dims(i);
    [unlabeledData, labels] = loadData(directory=datasetDir, contamination=eps, dimensions=p);

    kModel = AutoRbfKernel(unlabeledData, bandwidth="modifiedmean");
    solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
    
    hSubset = categorical(repmat("not in H", [height(unlabeledData) 1]), {'in H' 'not in H'});
    hSubset(solution.hsubsetIndices) = "in H";
    
    % Mahalanobis Distances
    % fig = figure();
    % mahalchart(hSubset, solution.rd, solution.cutoff);
    % saveas(fig, fullfile(imageDir, "AutoRbfKernel (Scaled Median)", sprintf("mahalanobis_distances_d%d_e0%.0f.png", p, eps*10)),'png');
    
    gaps{i, 4} = gap(hSubset, solution.rd);
end

%% Rbf (p^1/2)
for i = 1:length(dims)
    p = dims(i);
    [unlabeledData, labels] = loadData(directory=datasetDir, contamination=eps, dimensions=p);

    kModel = RbfKernel(sqrt(p));
    solution = kMRCD(kModel).runAlgorithm(unlabeledData, alpha);
    
    hSubset = categorical(repmat("not in H", [height(unlabeledData) 1]), {'in H' 'not in H'});
    hSubset(solution.hsubsetIndices) = "in H";
    
    % Mahalanobis Distances
    % fig = figure();
    % mahalchart(hSubset, solution.rd, solution.cutoff);
    % saveas(fig, fullfile(imageDir, "RbfKernel (p^1/2)", sprintf("mahalanobis_distances_d%d_e0%.0f.png", p, eps*10)),'png');
    
    gaps{i, 5} = gap(hSubset, solution.rd);
end

%% Gaps

fig = figure();
plot(gaps,"Dimensions",["AutoRbfKernel (Median)", "AutoRbfKernel (Scaled Median)", "AutoRbfKernel (Mean)", "AutoRbfKernel (Modified Mean)", "RbfKernel (p^1/2)"]);
ylabel("Relative Gap size");
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
    data = rZscores(data);
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
    g = g / range(distances);
end
