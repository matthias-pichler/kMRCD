function mahalchart(labels,distances,cutoff)
    arguments
        labels (:,1) categorical
        distances (:,1) double
        cutoff (1,1) double
    end

    assert(isequal(size(labels), size(distances)));

    % plot mahalanobis distances
    hold on;
    title('Robust Mahalanobis Distances');
    gscatter(1:length(labels), distances, labels)
    yline(cutoff);
    ylim([0 inf]);
    legend([string(categories(labels)); "Cutoff"])
    hold off;
end

