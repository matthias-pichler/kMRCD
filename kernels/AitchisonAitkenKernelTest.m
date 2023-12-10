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
            expected = [[(1-l)^2,   l^2]
                        [l^2,       (1-l)^2]];

            x = eye(2);
            kernel = AitchisonAitkenKernel(x, lambda=[l l]);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity2lambda(testCase)
            l1 = 0.75;
            l2 = 0.5;
            expected = [[(1-l1)*(1-l2), l1*l2]
                        [l1*l2,         (1-l1)*(1-l2)]];

            x = eye(2);
            kernel = AitchisonAitkenKernel(x, lambda=[l1 l2]);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity3(testCase)
            l = 0.75;
            expected = [[(1-l)^3,   l^2*(1-l),  l^2*(1-l)]
                        [l^2*(1-l), (1-l)^3,    l^2*(1-l)]
                        [l^2*(1-l), l^2*(1-l),  (1-l)^3]];

            x = eye(3);
            kernel = AitchisonAitkenKernel(x, lambda=[l l l]);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity3autolambda(testCase)
            l = 1/2 / (1+(3*(1/2-1/3)^2/(1/3*(1-1/3))));
            
            expected = [[(1-l)^3,   l^2*(1-l),  l^2*(1-l)]
                        [l^2*(1-l), (1-l)^3,    l^2*(1-l)]
                        [l^2*(1-l), l^2*(1-l),  (1-l)^3]];

            x = eye(3);
            kernel = AitchisonAitkenKernel(x);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected, AbsTol=0.0001);
        end

        function identity3lambda(testCase)
            l1 = 0.75;
            l2 = 0.5;
            l3 = 0.6;
            expected = [[(1-l1)*(1-l2)*(1-l3),  (1-l3)*l1*l2,           (1-l2)*l1*l3]
                        [(1-l3)*l1*l2,          (1-l1)*(1-l2)*(1-l3),   (1-l1)*l2*l3]
                        [(1-l2)*l1*l3,          (1-l1)*l2*l3,           (1-l1)*(1-l2)*(1-l3)]];

            x = eye(3);
            kernel = AitchisonAitkenKernel(x, lambda=[l1 l2 l3]);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function identity4(testCase)
            l = 0.75;
            expected = [[(1-l)^4,       l^2*(1-l)^2,    l^2*(1-l)^2,    l^2*(1-l)^2]
                        [l^2*(1-l)^2,   (1-l)^4,        l^2*(1-l)^2,    l^2*(1-l)^2]
                        [l^2*(1-l)^2,   l^2*(1-l)^2,    (1-l)^4,        l^2*(1-l)^2]
                        [l^2*(1-l)^2,   l^2*(1-l)^2,    l^2*(1-l)^2,    (1-l)^4]];

            x = eye(4);
            kernel = AitchisonAitkenKernel(x, lambda=[l l l l]);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end
    
        function tall(testCase)
            l = 0.75;
            expected = [[(1-l)^2,   l/2*l,      l/2*l,      l/2*(1-l)]
                        [l/2*l,     (1-l)^2,    l/2*(1-l),  l/2*l]
                        [l/2*l,     l/2*(1-l)   (1-l)^2,    (1-l)*l]
                        [l/2*(1-l), l/2*l,      (1-l)*l,    (1-l)^2]];

            x = [[1, 0]
                 [0, 1]
                 [2, 1]
                 [2, 0]];
            kernel = AitchisonAitkenKernel(x, lambda=[l l]);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected, AbsTol=0.001);
        end

        function fat(testCase)
            l = 0.75;
            expected = [[(1-l)^3,       l^2*(1-l)]
                        [l^2*(1-l), (1-l)^3]];

            x = [[1, 0, 0]
                 [0, 1, 0]];
            kernel = AitchisonAitkenKernel(x, lambda=[l l l]);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end
    end
    
end