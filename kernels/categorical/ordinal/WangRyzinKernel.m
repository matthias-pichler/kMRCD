classdef WangRyzinKernel < KernelModel
    
    % https://www.jstor.org/stable/2335831
    %
    %
    % k(x,y) =    PROD (1-l_i)  *  PROD 1/2 * (1-l_i) * l_i^|x_i-y_i|
    %           i:x_i = y_i     i:x_i != y_i
    %

    properties (GetAccess = public, SetAccess = private)
        lambda (1,:) double {mustBeInRange(lambda,0,1)}
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
            d = (1-l).^(~d_xy);

            r = 0.5 * (1-l) .* l.^abs(ZI - ZJ);
            r(~d_xy) = 1;

            d = prod(d,2) .* prod(r, 2);
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
            l = (c-1) ./ c;
        end
     end

    methods (Access = public)
        
        function this = WangRyzinKernel(x, NameValueArgs)
            arguments
                x double
                NameValueArgs.lambda (1,:) double = WangRyzinKernel.pluginbandwidth(x);
            end
            
            this.lambda = NameValueArgs.lambda;
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

