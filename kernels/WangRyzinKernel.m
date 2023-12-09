classdef WangRyzinKernel < handle
    
    % https://www.jstor.org/stable/2335831
    %
    %
    % k(x,y) =     PROD l_i  *  PROD 1/2 * l_i * (1-l_i)^|x_i-y_i|
    %           i:x_i = y_i  i:x_i != y_i
    %

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

            l = this.lambda;
            
            % differences between x_i, y_i
            d_xy = ZI ~= ZJ;

            % l^aggreements
            d = l.^(~d_xy);

            r = 0.5 * l .* (1-l).^abs(ZI - ZJ);
            r(~d_xy) = 1;

            d = prod(d,2) .* prod(r, 2);
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

