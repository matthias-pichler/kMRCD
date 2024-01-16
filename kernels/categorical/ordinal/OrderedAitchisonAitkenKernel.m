classdef OrderedAitchisonAitkenKernel < handle
    %
    %            n
    % k(x,y) = PROD binom(c_i,|x_i-y_i|)*l_i^|x_i-y_i|*(1-l_i)^(c_i - |x_i-y_i|)
    %           i=1
    %
    % c_i = number of categories for variable X_i

    properties (Access = private)
        categories (1,:) double
    end
    
    properties (GetAccess = public, SetAccess = private)
        lambda (1,:) double {mustBeInRange(lambda,0,0.5)}
    end

    methods (Access = private)
        function d = aitchisonaitkendist(this, ZI, ZJ)
            arguments
                this OrderedAitchisonAitkenKernel
                ZI (1,:) double
                ZJ double
            end

            l = this.lambda;
            c = this.categories;
            
            % differences between x_i, y_i
            s = abs(ZI - ZJ);

            C = repmat(c, [height(ZJ), 1]);

            % l^s * (1-l)^(c-s)
            d = l.^(s).*(1-l).^(C-s);
            
            nck = ones(size(d));
            for i=1:height(s)
                for j = 1:width(s)
                    nck(i,j) = nchoosek(c(j), s(i, j));
                end
            end
            
            d = nck .* d;

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

            numerator = n * cell2mat(cellfun(@(p){sum((1/numel(p)-p).^2)}, pmf));
            denominator = cell2mat(cellfun(@(p){sum(p.*(1-p))}, pmf));

            l = ((c-1)./c) ./ (1+(numerator./denominator));
        end
    
        function l = crossvalidatedbandwidth(x)
            arguments
                x double
            end

            n = height(x);

            numValues = 100;
            l_n = (1:numValues) / (2*numValues);

            best_W = 0;
            
            for i=1:numValues
                l_i = l_n(i);
                
                p = zeros(1, n);
                for j=1:n
                    D = x;
                    D(j,:) = [];
                    K = OrderedAitchisonAitkenKernel(D, lambda=l_i).compute(D, x(j,:));
                    p = mean(K);
                end

                W = prod(p);

                if W > best_W
                    best_W = W;
                    l = l_i;
                end
            end
        end
    end

    methods (Access = public)
        
        function this = OrderedAitchisonAitkenKernel(x, NameValueArgs)
            arguments
                x double
                NameValueArgs.lambda (1,:) double = OrderedAitchisonAitkenKernel.pluginbandwidth(x);
            end
            
            g = cellfun(@groupcounts, num2cell(x, 1), UniformOutput=false);

            this.categories = cellfun(@length, g);
            
            this.lambda = NameValueArgs.lambda;
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this OrderedAitchisonAitkenKernel
                Xtrain double
                Xtest double = Xtrain
            end
            
            K = pdist2(Xtrain, Xtest, @this.aitchisonaitkendist);
            
            assert(size(K, 1)==size(Xtrain, 1));
            assert(size(K, 2)==size(Xtest, 1));
        end
    end
end

