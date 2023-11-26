function si = separabilityindex(data,labels,NameValueArgs)
    arguments
        data double {mustBeNonempty}
        labels (:,1) {mustBeNonempty}
        NameValueArgs.Distance (1,:) char {mustBeMember(NameValueArgs.Distance,{'euclidean','cosine'})} = 'euclidean'
    end

    % find the nearest neighbor in the data, the first will be the point
    % itself and the second
    neighbors = knnsearch(data,data,K=2,Distance=NameValueArgs.Distance);

    idxs = neighbors(:,2);

    neighborLabels = labels(idxs);

    si = sum(neighborLabels == labels)/numel(labels);
end

