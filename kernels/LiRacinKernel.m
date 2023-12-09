classdef LiRacinKernel < handle
    %
    %
    % k(x,y) =   PROD l_i
    %         i:x_i != y_i

    properties (Access = public)
        lambda (1,:) double {mustBeInRange(lambda,0,1)}
    end

    methods (Access = private)
        function d = liracindist(this, ZI, ZJ)
            arguments
                this LiRacinKernel
                ZI (1,:) double
                ZJ double
            end

            l = this.lambda;
            
            % differences between x_i, y_i
            d_xy = ZI ~= ZJ;

            % l^disaggreements
            d = l.^d_xy;
            d = prod(d,2);
        end
    end
    
    methods (Access = public)
        
        function this = LiRacinKernel(x)
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
                this LiRacinKernel
                Xtrain double
                Xtest double = Xtrain
            end
            
            K = pdist2(Xtrain, Xtest, @this.liracindist);
            
            assert(size(K, 1)==size(Xtrain, 1));
            assert(size(K, 2)==size(Xtest, 1));
        end
    end
    
end

