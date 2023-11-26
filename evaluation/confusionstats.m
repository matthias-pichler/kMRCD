function result = confusionstats(cm) 
    arguments
        cm (:,:) {mustBeSquare}
    end

    true_negatives = cm(1,1);
    false_negatives = cm(2,1);
    
    true_positives = cm(2,2);
    false_positves = cm(1,2);
    
    n = sum(cm, "all");

    result = struct();

    result.accuracy = (true_positives + true_negatives)/n;
    result.precision = true_positives / (true_positives + false_positves);
    result.sensitivity = true_positives / (true_positives + false_negatives); % recall
    result.specificity = true_negatives / (true_negatives + false_positves);
    result.f1Score = 2 * true_positives / (2 * true_positives + false_positves + false_negatives);
end