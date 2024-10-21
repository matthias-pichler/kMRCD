fileDir = fileparts(which(mfilename));

inputFile = fullfile(fileDir, 'ssk.cpp');
outDir = fileDir;

mex(inputFile, ...
    '-v',...
    '-outdir', outDir, ...
    'CXXFLAGS=$CXXFLAGS -std=c++17mex -setup c++');
