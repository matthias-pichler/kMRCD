function objectiveplot(poc, zScores)
    arguments
        poc kMRCD
        zScores double {mustBeNonempty}
    end

    hSubsetRelativeSize = 0.5:0.05:1;
    objectiveValues = zeros(size(hSubsetRelativeSize));
    for i=1:numel(hSubsetRelativeSize)
        objectiveValues(i) = poc.runAlgorithm(zScores, hSubsetRelativeSize(i)).obj;
    end

    figure;
    plot(hSubsetRelativeSize, objectiveValues, 'o')
    title('kMRCD Objective Value vs h-Subset size')
    xlabel('h/n')
    ylabel('det(K)')
end

