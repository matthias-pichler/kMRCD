classdef LiRacinKernel < KernelModel
    %
    %
    % k(x,y) =   PROD l_i
    %         i:x_i != y_i

    properties (GetAccess = public, SetAccess = private)
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

    methods (Static, Access = private)
        function l = pluginbandwidth(x)
            arguments
                x double
            end
            
            n = height(x);

            [~, ~, gp] = cellfun(@groupcounts, num2cell(x, 1), UniformOutput=false);

            pmf = cellfun(@(c)c/100, gp, UniformOutput=false);

            numerator = n * cell2mat(cellfun(@(p){sum((1-p).^2)}, pmf));
            denominator = cell2mat(cellfun(@(p){sum(p.*(1-p))}, pmf));

            l = 1 ./ (1+(numerator./denominator));
        end
    end

    methods (Access = public)
        
        function this = LiRacinKernel(x, NameValueArgs)
            arguments
                x double
                NameValueArgs.lambda (1,:) double = LiRacinKernel.pluginbandwidth(x);
            end
            
            this.lambda = NameValueArgs.lambda;

            disp(['LiRacinKernel: Lambda = ' mat2str(this.lambda)]);
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

