function data = halfkernel(N1, N2, minx, r1, r2, noise, ratio)
% HALFKERNEL generates a non-elliptical 2D dataset
%  data = HALFKERNEL(N1, N2, minx, r1, r2, noise, ratio)
%
% Input
%   N1 (1,1) double {mustBePositive, mustBeInteger}
%       number of inliers
%   N2 (1,1) double {mustBePositive, mustBeInteger}
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
%   data (N1+N2,3) double
%       data(:,1) - x coordinate
%       data(:,2) - y coordinate
%       data(:,3) - class label
%       data(1:N1,3) = 1 inliers
%       data(N1+1:end,3) = 0 outliers

arguments
    N1 (1,1) double {mustBePositive, mustBeInteger}
    N2 (1,1) double {mustBePositive, mustBeInteger}
    minx (1,1) double
    r1 (1,1) double {mustBePositive}
    r2 (1,1) double {mustBePositive}
    noise (1,1) double {mustBePositive}
    ratio (1,1) double {mustBeInRange(ratio, 0, 1)}
end

phi1 = rand(N1,1) * pi;
inner = [minx + r1 * sin(phi1) - .5 * noise  + noise * rand(N1,1), r1 * ratio * cos(phi1) - .5 * noise + noise * rand(N1,1), ones(N1,1)];

phi2 = rand(N2,1) * pi;
outer = [minx + r2 * sin(phi2) - .5 * noise  + noise * rand(N2,1), r2 * ratio * cos(phi2) - .5 * noise  + noise * rand(N2,1), zeros(N2,1)];

data = [inner; outer];
end