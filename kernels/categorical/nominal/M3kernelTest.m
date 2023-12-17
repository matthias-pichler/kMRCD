classdef M3kernelTest < matlab.unittest.TestCase
    
    methods(TestClassSetup)
        % Shared setup for the entire test class
    end
    
    methods(TestMethodSetup)
        % Setup for each test
    end
    
    methods(Test)
        % Test methods
        
        function identity3(testCase)
            expected = [[1,   1/4, 1/4]
                        [1/4, 1  , 1/4]
                        [1/4, 1/4, 1  ]];
            x = eye(3);
            kernel = M3Kernel(x, alpha=1);

            res = kernel.compute(x);
            testCase.verifyEqual(res, expected, AbsTol=1/10000);
        end

        function tall(testCase)
            expected = [[1,     0,      0,      4/9]
                        [0,     1,      4/9,    0]
                        [0,     4/9     1,      1/2]
                        [4/9,   0,      1/2,    1]];

            x = [[1, 0]
                 [0, 1]
                 [2, 1]
                 [2, 0]];
            kernel = M3Kernel(x, alpha=1);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected, AbsTol=0.001);
        end

        function fat(testCase)
            expected = [[1, 0]
                        [0, 1]];

            x = [[1, 0, 0]
                 [0, 1, 0]];
            kernel = M3Kernel(x, alpha=1);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end
    end
    
end