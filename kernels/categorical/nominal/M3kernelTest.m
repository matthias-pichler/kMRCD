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

        function tall1(testCase)
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

            testCase.verifyEqual(res, expected, AbsTol=0.000001);
        end

        function tall2(testCase)
            expected = [[1    0   0.4 0]
                        [0    1   0   0]
                        [0.4  0   1   0]
                        [0    0   0   1]];

            x = [[7 3]
                 [5 5]
                 [2 3]
                 [8 4]];
            kernel = M3Kernel(x, alpha=1);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected, AbsTol=0.000001);
        end

        function tall3(testCase)
            expected = [[0.4 0    0.4]
                        [0   0.5  0]
                        [1   0    0.4]
                        [0   0.5  6/11]];

            x = [[7 3]
                 [5 5]
                 [2 3]
                 [8 4]];
            y = [[2 3]
                 [8 5]
                 [8 3]];

            kernel = M3Kernel(x, alpha=1);

            res = kernel.compute(x, y);

            testCase.verifyEqual(res, expected, AbsTol=0.000001);
        end

        function fat1(testCase)
            expected = [[1, 0]
                        [0, 1]];

            x = [[1, 0, 0]
                 [0, 1, 0]];
            kernel = M3Kernel(x, alpha=1);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end

        function fat2(testCase)
            expected = [1/2
                        4/15 
                        4/15];

            x = [[9     4     8     7]
                 [1     5     1     6]
                 [4     3     1     9]];
            
            y = [1      3     8     7];
            kernel = M3Kernel(x, alpha=1);

            res = kernel.compute(x, y);

            testCase.verifyEqual(res, expected);
        end

        function rand1(testCase)
            expected = [[1      0.2 0.2 0]
                         0.2    1   0.2 0
                         0.2    0.2 1   0
                         0      0   0   1];

            x = [[1     5     1     7]
                 [1     6     2     4]
                 [2     2     2     7]
                 [8     3     6     5]];
            kernel = M3Kernel(x, alpha=1);

            res = kernel.compute(x);

            testCase.verifyEqual(res, expected);
        end
    end
    
end