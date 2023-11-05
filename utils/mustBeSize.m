function mustBeSize(M, varargin)
    % MUSTBESIZE(M) validates that M has size sz.

    sz = cell2mat(varargin);

    if ~isequal(size(M), sz)
        eid = 'Size:notMatching';
        msg = sprintf('The matrix must be of size (%s)', num2str(sz, "%d "));
        error(eid,msg)
    end
end