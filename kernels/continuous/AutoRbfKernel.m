classdef AutoRbfKernel < RbfKernel
    methods (Static, Access = public)

        function sigma = medianBandwidth(x)
            arguments
                x double
            end

            distances = pdist(x, "squaredeuclidean");
            sigma = sqrt(median(distances));
        end

        function sigma = scaledmedianBandwidth(x)
            arguments
                x double
            end

            p = width(x);
            scale = sqrt(p);

            sigma = scale * AutoRbfKernel.medianBandwidth(x);
        end

        function sigma = meanBandwidth(x)
            arguments
                x double
            end

            % see Chaudhuri et al. 2017
            N = height(x);
            delta = sqrt(2) * 10^-6;

            s2 = var(x);

            numerator = 2 * N * sum(s2);
            denominator = (N-1) * log((N-1)/delta^2);

            sigma = sqrt( numerator / denominator );
        end

        function sigma = modifiedmeanBandwidth(x)
            arguments
                x double
            end

            % see Liao et al. 2018
            N = height(x);
            phi = 1 / log(N - 1);
            delta = -0.14818008* phi^4 + 0.284623624 * phi^3 - 0.252853808 * phi^2 + 0.159059498 * phi - 0.001381145;

            s2 = var(x);

            numerator = 2 * N * sum(s2);
            denominator = (N-1) * log((N-1)/delta^2);

            sigma = sqrt( numerator / denominator );
        end

    end
    
    methods (Access = public)
        
        function this = AutoRbfKernel(x, NameValueArgs)
            arguments
                x double
                NameValueArgs.bandwidth (1,1) string {mustBeMember(NameValueArgs.bandwidth, {'median' 'scaledmedian' 'mean' 'modifiedmean'})} = 'median';
            end

            bandwidth = 1;
            if strcmp(NameValueArgs.bandwidth, 'median')
                bandwidth = AutoRbfKernel.medianBandwidth(x);
            elseif strcmp(NameValueArgs.bandwidth, 'scaledmedian')
                bandwidth = AutoRbfKernel.scaledmedianBandwidth(x);
            elseif strcmp(NameValueArgs.bandwidth, 'mean')
                bandwidth = AutoRbfKernel.meanBandwidth(x);
            elseif strcmp(NameValueArgs.bandwidth, 'modifiedmean')
                bandwidth = AutoRbfKernel.modifiedmeanBandwidth(x);
            end

            this@RbfKernel(bandwidth);
        end
    end
    
end

