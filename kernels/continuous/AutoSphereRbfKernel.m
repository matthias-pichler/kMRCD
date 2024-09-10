classdef AutoSphereRbfKernel < SphereRbfKernel

    methods (Static, Access = protected)
        function sigma = medianBandwidth(x)
            arguments
                x double
            end

            % 2(1- <x,y>) = ||x-y||^2 on the unit sphere
            distances = 2 * pdist(x, "cosine");
            sigma = sqrt(median(distances));
        end
    end

    methods (Access = public)
        
        function this = AutoSphereRbfKernel(x, NameValueArgs)
            arguments
                x double
                NameValueArgs.bandwidth (1,1) string {mustBeMember(NameValueArgs.bandwidth, {'median' 'mean' 'modifiedmean'})} = 'median';
            end

            bandwidth = 1;
            if strcmp(NameValueArgs.bandwidth, 'median')
                bandwidth = AutoSphereRbfKernel.medianBandwidth(x);
            elseif strcmp(NameValueArgs.bandwidth, 'mean')
                bandwidth = AutoRbfKernel.meanBandwidth(x);
            elseif strcmp(NameValueArgs.bandwidth, 'modifiedmean')
                bandwidth = AutoRbfKernel.modifiedmeanBandwidth(x);
            end

            this@SphereRbfKernel(bandwidth);
        end
    end
    
end

