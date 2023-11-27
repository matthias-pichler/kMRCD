classdef AutoSphereRbfKernel < handle
    % K(x,y) = k(x*y) = exp(-1/s*2(1-x*y))
    % if ||x|| = ||y|| = 1 then
    % ||x -y||^2 = ||x||^2 - 2<x,y> + ||y||^2
    %            = 1 - 2<x,y> +1 = 2 - 2<x,y> = 2(1 - <x-y>)

    properties (Access = public)
        sigma;
    end
    
    methods (Access = public)
        
        function this = AutoSphereRbfKernel(x)
            arguments
                x double
            end
            
            % TODO: is this still the best way to estimate sigma?
            % 2(1- <x,y>) = ||x-y||^2 on the unit sphere
            distances = 2 * pdist(x, "cosine");
            this.sigma = sqrt(median(distances));
            disp(['AutoSphereRbfKernel: Sigma = ' mat2str(this.sigma)]);
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this
                Xtrain double
                Xtest double = Xtrain
            end
            
            
            K = 1 - (Xtrain * Xtest');
            K = exp(-(2/this.sigma^2) * K);
        end
    end
    
end

