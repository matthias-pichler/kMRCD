classdef ClusterContamination < handle
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    methods(Access = public)
        function xx = generateContamination(~, m, p, r, tLoc, tCor)
            %GENERATECONTAMINATION Generate contaminated data
            %   xx = generateContamination(contaminationDegree, p, replacement, tLocation, Sigma_outlier)
            %
            %   Input
            %   contaminationDegree (1,1) double {mustBePositive, mustBeInteger}
            %       Number of outliers to be generated
            %   p (1,1) double {mustBePositive, mustBeInteger}
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
                m (1,1) double {mustBePositive, mustBeInteger}
                p (1,1) double {mustBePositive, mustBeInteger}
                r (:,1) double {mustBeSize(r, p, 1)}
                tLoc (:,1) double {mustBeSize(tLoc, p, 1)}
                tCor (:,:) double {mustBeSize(tCor, p, p)}
            end
            
            xx = mvnrnd((r + tLoc)', tCor, m);
        end
    end
end
