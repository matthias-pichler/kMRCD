classdef OrderedLiRacinKernel < handle
    %
    %            n
    % k(x,y) = PROD l_i^|x_i-y_i|
    %           i=1
    %
    % d(x,y) = number of disaggreements

    properties (Access = public)
        lambda (1,:) double {mustBeInRange(lambda,0,1)}
    end

    methods (Access = private)
        function d = liracindist(this, ZI, ZJ)
            arguments
                this OrderedLiRacinKernel
                ZI (1,:) double
                ZJ double
            end

            l = this.lambda;
            
            % l^disaggreements
            d = l.^abs(ZI - ZJ);

            d = prod(d,2);
        end
    end
    
    methods (Access = public)
        
        function this = OrderedLiRacinKernel(x)
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
                this OrderedLiRacinKernel
                Xtrain double
                Xtest double = Xtrain
            end
            
            K = pdist2(Xtrain, Xtest, @this.liracindist);
            
            assert(size(K, 1)==size(Xtrain, 1));
            assert(size(K, 2)==size(Xtest, 1));
        end
    end
    
end

