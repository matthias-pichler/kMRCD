classdef K1Kernel < handle
    
% https://upcommons.upc.edu/bitstream/handle/2099.1/24508/99930.pdf
% 
%                   n
% k_1(x,y) = 1/n * SUM k_1U(x_i,y_i)
%                  i=1
%
% k_1U(x,y) = h_a(P(x)) if x=y else 0
% 
% h_a(z) = (1-z^a)^(1/a)

    properties (Access = private)
        columnCategories cell
        columnPmf cell
    end
    
    methods (Static, Access = private)
        function r = h(z, alpha)
            arguments
                z double
                alpha (1,1) double {mustBePositive} = 1
            end

            r = (1-z.^alpha).^(1/alpha);

        end
    end

    methods (Access = private)
        function d = k1dist(this, ZI, ZJ)
            arguments
                this K1Kernel
                ZI (1,:) double
                ZJ double
            end

            P_x = zeros(size(ZI));
            n = numel(ZI);

            for columnIndex=1:n
                currentElement = ZI(columnIndex);
                categoryIndex = this.columnCategories{columnIndex} == currentElement;
                pmf = this.columnPmf{columnIndex}(categoryIndex);
                P_x(columnIndex) = pmf;
            end

            mask = ZI == ZJ;
            summands = K1Kernel.h(P_x);
            
            d = (mask * summands')/n;
        end
    end

    methods (Access = public)
        
        function this = K1Kernel(x)
            arguments
                x double
            end
            
            [~, gr, gp] = cellfun(@groupcounts, num2cell(x, 1), UniformOutput=false);

            this.columnCategories = gr;
            this.columnPmf = cellfun(@(c)c/100, gp, UniformOutput=false);
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this
                Xtrain double
                Xtest double = Xtrain
            end
            
            K = pdist2(Xtrain, Xtest, @this.k1dist);
            
            assert(size(K, 1)==size(Xtrain, 1));
            assert(size(K, 2)==size(Xtest, 1));
        end
    end
    
end

