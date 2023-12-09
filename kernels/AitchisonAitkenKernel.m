classdef AitchisonAitkenKernel < handle
    %
    %
    % k(x,y) =   PROD l_i  * PROD (1-l_i)/(c_i-1)
    %         i:x_i=y_i    i:x_i != y_i
    %
    % c_i = number of categories for variable X_i

    properties (Access = private)
        categories (1,:) double
    end
    
    properties (Access = public)
        lambda (1,:) double {mustBeInRange(lambda,0.5,1)}
    end

    methods (Access = private)
        function d = aitchisonaitkendist(this, ZI, ZJ)
            arguments
                this AitchisonAitkenKernel
                ZI (1,:) double
                ZJ double
            end

            l = this.lambda;
            c = this.categories;
            
            % differences between x_i, y_i
            d_xy = ZI ~= ZJ;

            % l^aggreements
            d = l.^(~d_xy);

            r = (1-l)./(c'-1);
            r(~d_xy) = 1;

            d = prod(d,2) .* prod(r, 2);
        end
    end
    
    methods (Access = public)
        
        function this = AitchisonAitkenKernel(x)
            arguments
                x double
            end
            
            g = cellfun(@groupcounts, num2cell(x, 1), UniformOutput=false);

            this.categories = cellfun(@length, g);
            
            % TODO: calculate lambda
            this.lambda = 1./this.categories;
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this AitchisonAitkenKernel
                Xtrain double
                Xtest double = Xtrain
            end
            
            K = pdist2(Xtrain, Xtest, @this.aitchisonaitkendist);
            
            assert(size(K, 1)==size(Xtrain, 1));
            assert(size(K, 2)==size(Xtest, 1));
        end
    end
end

