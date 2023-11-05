classdef NewDataModel < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        CorrelationType;
        ContaminationType;
    end
    
    methods(Access = private)
        function Sigma = generateCormatrix(~, p, correl)
            % generates a correlation matrix; correl = 0.9 is the default choice
            % Note: -correl is used as base value (!)
            
            arguments
                ~
                p (1,1) double {mustBeInteger, mustBePositive}
                correl (1,1) double {mustBeInRange(correl, 0, 1)} = 0.9
            end
            
            columns = repmat((1:p),p,1);
            rows    = repmat((1:p)',1,p);
            Sigma   = -correl*ones(p);
            Sigma   = Sigma.^(abs(columns - rows)) ;
        end
    end
    
    methods(Access = public)
        function this = NewDataModel(CorrelationType, ContaminationType)
            this.CorrelationType = CorrelationType;
            this.ContaminationType = ContaminationType;
        end
        
        
        function [samples, tCorrelation, tLocation, cindices] = generateDataset(this, n,  p, eps, k)
            % GENERATEDATASET Generates a dataset with n samples and p dimensions
            % and n*eps outliers.
            %
            %   [samples, tCorrelation, tLocation, cindices] = generateDataset(this, n,  p, eps, k)
            %
            % Input
            %   n (1,1) double {mustBeInteger, mustBePositive}
            %       Number of samples
            %   p (1,1) double {mustBeInteger, mustBePositive}
            %       Number of dimensions
            %   eps (1,1) double {mustBeInRange(eps, 0, 1)}
            %       Percentage of outliers
            %   k (1,1) double {mustBeInteger, mustBePositive}
            %       Number of standard deviations of the outlier
            %
            % Output
            %   samples (n,p) double
            %       Generated dataset
            %   tCorrelation (p,p) double
            %       Correlation matrix used to generate the dataset
            %   tLocation (p,1) double
            %       Location vector used to generate the dataset
            %   cindices (1,contaminationDegree) double
            %       Indices of the outliers in the dataset
            
            arguments
                this
                n (1,1) double {mustBeInteger, mustBePositive}
                p (1,1) double {mustBeInteger, mustBePositive}
                eps (1,1) double {mustBeInRange(eps, 0, 1)}
                k (1,1) double {mustBeInteger, mustBePositive}
            end
            
            contaminationDegree = floor(n * eps);
            
            %	Generate correlation matrix
            tCorrelation = this.CorrelationType.generateCorr(p);
            tLocation = this.CorrelationType.generateLocation(p);
            
            %	Generate multivariate normal data with location zero and correlation;
            samples = mvnrnd(tLocation, tCorrelation, n);
            
            %   Generate contamination
            [U,~,~] = svd(tCorrelation);
            replacement = U(:,end);
            
            delta = replacement - tLocation;
            smd = sqrt((delta' * inv(tCorrelation) * delta));
            replacement = replacement * (k/smd);
            
            % Sigma contamination
            %             Sigma_outlier = k*this.generateCormatrix(p,0.95);
            %             replacement = tLocation;
            Sigma_outlier = tCorrelation;
            
            contamination = this.ContaminationType.generateContamination(contaminationDegree, p, replacement, tLocation, Sigma_outlier);
            
            cindices = randperm(n, contaminationDegree);
            samples(cindices, :) = contamination;
        end
    end
    
end

