function mustBeSquare(M)
    % MUSTBESQUARE(M) validates that M is a square matrix.

    if ~isequal(size(M,1),size(M,2))
        eid = 'Size:notSquare';
        msg = 'The matrix must be square.';
        error(eid,msg)
    end
end