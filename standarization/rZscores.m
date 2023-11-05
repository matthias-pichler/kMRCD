function [x,mu,sigma] = rZscores( x )
    %RZSCORES Normalizes the (n,p) matrix x by estimating the mean and standard
    % deviation of each feature (column) using the univariate MCD estimator and then
    % subtracting the mean and dividing by the standard deviation
    %
    % Input
    %   x (:, :) double
    %       The data matrix
    %
    % Output
    %   x (:, :) double
    %       The normalized data matrix
    %   mu (1, p) double
    %       The estimated mean of each feature
    %   sigma (1, p) double
    %       The estimated standard deviation of each feature

    arguments
        x (:, :) double
    end

    [n, p] = size(x);
    mu = nan(1, p);
    sigma = nan(1, p);
    for featureIndex=1:p
        [tmcd,smcd] = unimcd(x(~isnan(x(:, featureIndex)), featureIndex), ceil(n*0.5));
        mu(featureIndex) = tmcd;
        sigma(featureIndex) = smcd;
        %        assert(smcd>1e-10);
    end
    mask = (sigma<1e-12) | (isnan(sigma));
    sigma(mask) = 1;
    mu(isnan(mu)) = 0;
    x = (x - repmat(mu, n, 1)) ./ repmat(sigma, n, 1);
end

