classdef ClusterContamination < handle
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    methods(Access = public)
        function xx = generateContamination(~, outliers, dimension, replacement, tLoc, tCor)
            %GENERATECONTAMINATION Generate contaminated data
            %   xx = generateContamination(contaminationDegree, p, replacement, tLocation, Sigma_outlier)
            %
            %   Input
            %   outliers (1,1) double {mustBePositive, mustBeInteger}
            %       Number of outliers to be generated
            %   dimension (1,1) double {mustBePositive, mustBeInteger}
            %       Dimension of the data
            %   replacement (1,1) logical
            %       If true, the outliers are generated with replacement
            %   tLocation (p,1) double
            %       Location of the outliers
            %   Sigma_outlier (p,p) double
            %       Covariance matrix of the outliers
            %
            %   Output
            %   xx (m, p) double
            %       Contaminated data
            
            arguments
                ~
                outliers (1,1) double {mustBeNonnegative, mustBeInteger}
                dimension (1,1) double {mustBePositive, mustBeInteger}
                replacement (:,1) double {mustBeSize(replacement, dimension, 1)}
                tLoc (:,1) double {mustBeSize(tLoc, dimension, 1)}
                tCor (:,:) double {mustBeSize(tCor, dimension, dimension)}
            end

            xx = mvnrnd((replacement + tLoc)', tCor, outliers);
        end
    end
end
