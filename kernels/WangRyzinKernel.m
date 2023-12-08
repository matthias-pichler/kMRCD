classdef WangRyzinKernel < handle
    
    % https://www.jstor.org/stable/2335831
    %
    %
    % k(x,y) = l^(n-d(x,y)) * PROD 1/2 * l * (1-l)^|x_i-y_i|
    %                      i:x_i != y_i
    %
    % d(x,y) = number of disaggreements
    % c_i = number of categories for variable X_i

    properties (Access = public)
        lambda (1,:) double {mustBeInRange(lambda,0.5,1)}
    end

    methods (Access = private)
        function d = wangryzindist(this, ZI, ZJ)
            arguments
                this WangRyzinKernel
                ZI (1,:) double
                ZJ double
            end

            n = numel(ZI);
            l = this.lambda';
            
            % differences between x_i, y_i
            d_xy = ZI ~= ZJ;

            % l^aggreements
            d = l.^(n-sum(d_xy, 2));

            r = 0.5 * l .* (1-l).^abs(ZI - ZJ);
            r(~d_xy) = 1;

            d = d .* prod(r, 2);
        end
    end
    
    methods (Access = public)
        
        function this = WangRyzinKernel(x)
            arguments
                x double
            end
            
            g = cellfun(@groupcounts, num2cell(x, 1), UniformOutput=false);

            categories = cellfun(@length, g);
            
            % TODO: calculate lambda
            this.lambda = 1./categories;
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this WangRyzinKernel
                Xtrain double
                Xtest double = Xtrain
            end
            
            K = pdist2(Xtrain, Xtest, @this.wangryzindist);
            
            assert(size(K, 1)==size(Xtrain, 1));
            assert(size(K, 2)==size(Xtest, 1));
        end
    end
    
end

