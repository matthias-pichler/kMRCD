classdef K1KernelTest < matlab.unittest.TestCase
    
    methods(TestClassSetup)
        % Shared setup for the entire test class
    end
    
    methods(TestMethodSetup)
        % Setup for each test
    end
    
    methods(Test)
        % Test methods
        
        function identity2(testCase)
            expected = [[1/2,0]
                        [0,1/2]];
            x = eye(2);
            kernel = K1Kernel(x, alpha=1);

            res = kernel.compute(x);
            testCase.verifyEqual(res, expected, AbsTol=1/10000);
        end

        function identity3(testCase)
            expected = [[4/9,1/9,1/9]
                        [1/9,4/9,1/9]
                        [1/9,1/9,4/9]];
            x = eye(3);
            kernel = K1Kernel(x, alpha=1);

            res = kernel.compute(x);
            testCase.verifyEqual(res, expected, AbsTol=1/10000);
        end
        
        function tall(testCase)
            expected = [[5/8,   0,      0,      1/4]
                        [0,     5/8,    1/4,    0]
                        [0,     1/4     1/2,    1/4]
                        [1/4,   0,      1/4,    1/2]];

            x = [[1, 0]
                 [0, 1]
                 [2, 1]
                 [2, 0]];
            kernel = K1Kernel(x, alpha=1);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected, AbsTol=0.001);
        end

        function fat(testCase)
            expected = [[1/3,   0]
                        [0,     1/3]];

            x = [[1, 0, 0]
                 [0, 1, 0]];
            kernel = K1Kernel(x, alpha=1);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end
    end
    
end