classdef AitchisonAitkenKernelTest < matlab.unittest.TestCase
    
    methods(TestClassSetup)
        % Shared setup for the entire test class
    end
    
    methods(TestMethodSetup)
        % Setup for each test
    end
    
    methods(Test)
        % Test methods
        
        function identity2(testCase)
            l = 0.75;
            expected = [[l^2,      (1-l)^2]
                        [(1-l)^2,  l^2]];

            x = eye(2);
            kernel = AitchisonAitkenKernel(x);
            kernel.lambda = [l l];

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity2lambda(testCase)
            l1 = 0.75;
            l2 = 0.5;
            expected = [[l1*l2,         (1-l1)*(1-l2)]
                        [(1-l1)*(1-l2), l1*l2]];

            x = eye(2);
            kernel = AitchisonAitkenKernel(x);
            kernel.lambda = [l1 l2];

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity3(testCase)
            l = 0.75;
            expected = [[l^3,       l*(1-l)^2,  l*(1-l)^2]
                        [l*(1-l)^2, l^3,        l*(1-l)^2]
                        [l*(1-l)^2, l*(1-l)^2,  l^3]];

            x = eye(3);
            kernel = AitchisonAitkenKernel(x);
            kernel.lambda = [l l l];

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity3lambda(testCase)
            l1 = 0.75;
            l2 = 0.5;
            l3 = 0.6;
            expected = [[l1*l2*l3,          l3*(1-l1)*(1-l2),   l2*(1-l1)*(1-l3)]
                        [l3*(1-l1)*(1-l2),  l1*l2*l3,           l1*(1-l2)*(1-l3)]
                        [l2*(1-l1)*(1-l3),  l1*(1-l2)*(1-l3),   l1*l2*l3]];

            x = eye(3);
            kernel = AitchisonAitkenKernel(x);
            kernel.lambda = [l1 l2 l3];

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity4(testCase)
            l = 0.75;
            expected = [[l^4,           l^2*(1-l)^2,    l^2*(1-l)^2,    l^2*(1-l)^2]
                        [l^2*(1-l)^2,   l^4,            l^2*(1-l)^2,    l^2*(1-l)^2]
                        [l^2*(1-l)^2,   l^2*(1-l)^2,    l^4,            l^2*(1-l)^2]
                        [l^2*(1-l)^2,   l^2*(1-l)^2,    l^2*(1-l)^2,    l^4]];

            x = eye(4);
            kernel = AitchisonAitkenKernel(x);
            kernel.lambda = [l l l l];

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end
    end
    
end