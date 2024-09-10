classdef AutoRbfKernel < RbfKernel
    methods (Static, Access = public)

        function sigma = medianBandwidth(x)
            arguments
                x double
            end

            distances = pdist(x, "squaredeuclidean");
            sigma = sqrt(median(distances));
        end

        function sigma = meanBandwidth(x)
            arguments
                x double
            end

            % see Chaudhuri et al. 2017
            N = height(x);
            delta = sqrt(2) * 10^-6;

            s2 = var(x);

            sigma = sqrt( 2 * N * sum(s2)/((N-1) * log((N-1)/delta^2)) );
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

            sigma = sqrt( 2 * N * sum(s2)/((N-1) * log((N-1)/delta^2)) );
        end

    end
    
    methods (Access = public)
        
        function this = AutoRbfKernel(x, NameValueArgs)
            arguments
                x double
                NameValueArgs.bandwidth (1,1) string {mustBeMember(NameValueArgs.bandwidth, {'median' 'mean' 'modifiedmean'})} = 'median';
            end

            bandwidth = 1;
            if strcmp(NameValueArgs.bandwidth, 'median')
                bandwidth = AutoRbfKernel.medianBandwidth(x);
            elseif strcmp(NameValueArgs.bandwidth, 'mean')
                bandwidth = AutoRbfKernel.meanBandwidth(x);
            elseif strcmp(NameValueArgs.bandwidth, 'modifiedmean')
                bandwidth = AutoRbfKernel.modifiedmeanBandwidth(x);
            end

            this@RbfKernel(bandwidth);
        end
    end
    
end

