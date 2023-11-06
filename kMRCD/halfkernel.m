function data = halfkernel(numInliers, numOutliers, minx, r1, r2, noise, ratio)
% HALFKERNEL generates a non-elliptical 2D dataset
%  data = HALFKERNEL(N1, numOutliers, minx, r1, r2, noise, ratio)
%
% Input
%   numInliers (1,1) double {mustBePositive, mustBeInteger}
%       number of inliers
%   numOutliers (1,1) double {mustBePositive, mustBeInteger}
%       number of outliers
%   minx (1,1) double
%       minimum x coordinate
%   r1 (1,1) double {mustBePositive}
%       radius of inner circle
%   r2 (1,1) double {mustBePositive}
%       radius of outer circle
%   noise (1,1) double {mustBePositive}
%       noise level
%   ratio (1,1) double {mustBeInRange(ratio, 0, 1)}
%       size ration between x and y span (ratio ~ (max(y)-min(y))/(max(x)-min(x))
%
% Output
%   data (N1+numOutliers,3) double
%       data(:,1) - x coordinate
%       data(:,2) - y coordinate
%       data(:,3) - class label
%       data(1:N1,3) = 1 inliers
%       data(N1+1:end,3) = 0 outliers

arguments
    numInliers (1,1) double {mustBePositive, mustBeInteger}
    numOutliers (1,1) double {mustBePositive, mustBeInteger}
    minx (1,1) double
    r1 (1,1) double {mustBePositive}
    r2 (1,1) double {mustBePositive}
    noise (1,1) double {mustBePositive}
    ratio (1,1) double {mustBeInRange(ratio, 0, 1)}
end

phi1 = rand(numInliers,1) * pi;
inner = [minx + r1 * sin(phi1) - .5 * noise  + noise * rand(numInliers,1), r1 * ratio * cos(phi1) - .5 * noise + noise * rand(numInliers,1), ones(numInliers,1)];

phi2 = rand(numOutliers,1) * pi;
outer = [minx + r2 * sin(phi2) - .5 * noise  + noise * rand(numOutliers,1), r2 * ratio * cos(phi2) - .5 * noise  + noise * rand(numOutliers,1), zeros(numOutliers,1)];

data = [inner; outer];
end