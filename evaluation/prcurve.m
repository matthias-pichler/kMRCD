function auc = prcurve(labels,scores, postiveClass, NameValueArgs)
    arguments
        labels (:,1) {mustBeNonempty}
        scores (:,:) double {mustBeNonempty}
        postiveClass
        NameValueArgs.DisplayName
        NameValueArgs.Color
    end

    [X,Y,~,auc] = perfcurve(labels,scores,postiveClass, xCrit='reca', yCrit='prec');

    plot(X,Y, Color=NameValueArgs.Color, ...
            DisplayName = sprintf('%s (AUC=%0.4f)', NameValueArgs.DisplayName, auc));
    ylim([0 1]);
end

