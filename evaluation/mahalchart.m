function mahalchart(labels,distances,cutoff)
    arguments
        labels (:,1) categorical
        distances (:,1) double
        cutoff (1,1) double
    end

    assert(isequal(size(labels), size(distances)))

    inlierIndices = find(labels == "inlier");
    outlierIndices = find(labels == "outlier");

    % plot mahalanobis distances
    hold on;
    title('Robust Mahalanobis Distances');
    plot(inlierIndices, distances(inlierIndices), '.', Color='green');
    plot(outlierIndices, distances(outlierIndices), '.', Color='red');
    yline(cutoff);
    hold off;
end
