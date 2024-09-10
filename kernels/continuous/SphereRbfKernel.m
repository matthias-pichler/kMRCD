classdef SphereRbfKernel < RbfKernel
    % K(x,y) = k(x*y) = exp(-1/s*2(1-x*y))
    % if ||x|| = ||y|| = 1 then
    % ||x -y||^2 = ||x||^2 - 2<x,y> + ||y||^2
    %            = 1 - 2<x,y> +1 = 2 - 2<x,y> = 2(1 - <x-y>)
    
    methods (Access = public)
        function this = SphereRbfKernel(bandwidth)
            arguments
                bandwidth (1,1) double {mustBePositive}
            end
            
            this@RbfKernel(bandwidth);
        end
        
        function K = compute(this, Xtrain, Xtest)
            arguments
                this SphereRbfKernel
                Xtrain double
                Xtest double = Xtrain
            end
            
            
            K = 1 - (Xtrain * Xtest');
            K = exp(-(2/this.sigma^2) * K);
        end
    end
    
end

