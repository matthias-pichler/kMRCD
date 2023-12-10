classdef OrderedLiRacinKernel < handle
    %
    %            n
    % k(x,y) = PROD l_i^|x_i-y_i|
    %           i=1
    %

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
            
            % l^|x_i-y_i|
            d = l.^abs(ZI - ZJ);

            d = prod(d,2);
        end
    end
    
    methods (Static, Access = private)
        function l = pluginbandwidth(x)
            arguments
                x double
            end
            
            n = height(x);

            [~, gr, gp] = cellfun(@groupcounts, num2cell(x, 1), UniformOutput=false);

            c = cellfun(@length, gr);
            pmf = cellfun(@(c)c/100, gp, UniformOutput=false);

            % TODO: find propper calculation
            l = 1 / c;
        end
     end

    methods (Access = public)
        
        function this = OrderedLiRacinKernel(x, NameValueArgs)
            arguments
                x double
                NameValueArgs.lambda (1,:) double = OrderedLiRacinKernel.pluginbandwidth(x);
            end
            
            this.lambda = NameValueArgs.lambda;
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

