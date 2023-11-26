function result = confusionstats(cm) 
    % CONFUSIONSTATS(cm) calculate accuracy, precision, sensitivity,
    % specificity and f1Score from a confusion matrix
    % 
    % Input
    %   cm (2, 2) double
    %       A two by two confusion matrix ot the form:
    %       
    %       tp | fn
    %       -------
    %       fp | tn
    %
    % Output
    %   T struct
    %       The normalized data matrix
    arguments
        cm (2,2) double {mustBeSquare}
    end

    true_positives = cm(1,1);
    false_positves = cm(2,1);

    true_negatives = cm(2,2);
    false_negatives = cm(1,2);
    
    
    n = sum(cm, "all");

    result = struct();

    result.accuracy = (true_positives + true_negatives)/n;
    result.precision = true_positives / (true_positives + false_positves);
    result.sensitivity = true_positives / (true_positives + false_negatives); % recall
    result.specificity = true_negatives / (true_negatives + false_positves);
    result.f1Score = 2 * true_positives / (2 * true_positives + false_positves + false_negatives);
end