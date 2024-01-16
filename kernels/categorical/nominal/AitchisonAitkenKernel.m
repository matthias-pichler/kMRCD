classdef AitchisonAitkenKernel < handle
    %
    %
    % k(x,y) =   PROD (1-l_i)  * PROD l_i/(c_i-1)
    %         i:x_i=y_i      i:x_i != y_i
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
                this AitchisonAitkenKernel
                ZI (1,:) double
                ZJ double
            end

            l = this.lambda;
            c = this.categories;
            
            % differences between x_i, y_i
            d_xy = ZI ~= ZJ;

            % (1-l)^aggreements
            d = (1-l).^(~d_xy);
            
            r = repmat(l./(c-1), [height(ZJ), 1]);
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

            c = AitchisonAitkenKernel.cats(x);
            pmf = AitchisonAitkenKernel.pmf(x);

            numerator = n * cell2mat(cellfun(@(p){sum((1/numel(p)-p).^2)}, pmf));
            denominator = cell2mat(cellfun(@(p){sum(p.*(1-p))}, pmf));

            l = ((c-1)./c) ./ (1+(numerator./denominator));

            disp(['AitchisonAitkenKernel: Lambda = ' mat2str(l)]);
        end

        function l = crossvalidatedbandwidth(x)
            arguments
                x double
            end

            [n, m] = size(x);

            numValues = 100;
            best_W = -Inf;

            l = repmat(0.25, 1, m);
            
            for i=1:numValues
                l_i = repmat(i / (2*numValues), 1, m);
                
                p = zeros(1, n);
                for j=1:n
                    D = x;
                    D(j,:) = [];
                    K = AitchisonAitkenKernel(x, lambda=l_i).compute(D, x(j,:));
                    
                    p(j) = mean(K);
                end

                W = prod(p);

                if W > best_W
                    best_W = W;
                    l = l_i;
                end
            end

            disp(['AitchisonAitkenKernel: Lambda = ' mat2str(l)]);
        end

        function c = cats(x)
            arguments
                x double
            end

            g = cellfun(@groupcounts, num2cell(x, 1), UniformOutput=false);
            c = cellfun(@length, g);
        end

        function p = pmf(x)
            arguments
                x double
            end

            [~, ~, gp] = cellfun(@groupcounts, num2cell(x, 1), UniformOutput=false);
            p = cellfun(@(c)c/100, gp, UniformOutput=false);
        end
    end

    methods (Access = public)
        
        function this = AitchisonAitkenKernel(x, NameValueArgs)
            arguments
                x double
                NameValueArgs.lambda (1,:) double = AitchisonAitkenKernel.pluginbandwidth(x);
                NameValueArgs.categories (1,:) double = AitchisonAitkenKernel.cats(x);
            end
            
            this.categories = NameValueArgs.categories;
            this.lambda = NameValueArgs.lambda;
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

