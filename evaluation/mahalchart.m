function mahalchart(labels,distances,cutoff)
    arguments
        labels (:,1) categorical
        distances (:,1) double
        cutoff (1,1) double
    end

    assert(isequal(size(labels), size(distances)));

    minValue = min([distances; cutoff]);
    maxValue = max([distances; cutoff]);
    valueRange = maxValue - minValue;
    plotHeight = maxValue + 0.07 * valueRange;

    % plot mahalanobis distances
    hold on;
    title('Robust Mahalanobis Distances');
    gscatter(1:length(labels), distances, labels)
    yline(cutoff);
    ylim([0 plotHeight]);
    legend([string(categories(labels)); "Cutoff"])
    hold off;
end

