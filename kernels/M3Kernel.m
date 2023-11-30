classdef M3Kernel < handle
    
% https://upcommons.upc.edu/bitstream/handle/2099.1/24508/99930.pdf
% 
%                                 n
% m3(x,y) =  SUM (h_a(P(x_i))) / SUM (h_a(P(x_i)) + h_a(P(y_i)))
%          i:x_i=y_i             i=1
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
        function d = m3distdenom(this, x, y)
            %  n
            % SUM h_a(P(x_i)) + h_a(P(y_i))
            % i=1
            arguments
                this M3Kernel
                x (1,:) double
                y (1,:) double
            end


            n = numel(x);

            P_x = zeros(size(x));
            P_y = zeros(size(y));

            for columnIndex=1:n
                currentElement = x(columnIndex);
                categoryIndex = this.columnCategories{columnIndex} == currentElement;
                pmf = this.columnPmf{columnIndex}(categoryIndex);
                P_x(columnIndex) = pmf;

                currentElement = y(columnIndex);
                categoryIndex = this.columnCategories{columnIndex} == currentElement;
                pmf = this.columnPmf{columnIndex}(categoryIndex);
                P_y(columnIndex) = pmf;
            end

            d = sum(M3Kernel.h(P_x) + M3Kernel.h(P_y));

        end

        function d = m3dist(this, ZI, ZJ)
            arguments
                this M3Kernel
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
            summands = M3Kernel.h(P_x);

            denom = cellfun(@(y)this.m3distdenom(ZI,y), num2cell(ZJ,2));
            
            d = 2 * (mask * summands')./denom;
        end
    end

  

    methods (Access = public)
        
        function this = M3Kernel(x)
            arguments
                x double
            end
            
             [~, gr, gp] = cellfun(@groupcounts, num2cell(x, 1), UniformOutput=false);

            this.columnCategories = gr;
            this.columnPmf = cellfun(@(c)c/100, gp, UniformOutput=false);
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this M3Kernel
                Xtrain double
                Xtest double = Xtrain
            end
            
            K = pdist2(Xtrain, Xtest, @this.m3dist);

            assert(size(K, 1)==size(Xtrain, 1));
            assert(size(K, 2)==size(Xtest, 1));
        end
    end
    
end

