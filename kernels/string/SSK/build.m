fileDir = fileparts(which(mfilename));

inputFile = fullfile(fileDir, 'ssk.cpp');
outDir = fileDir;

cxxFlags = [
    "CXXFLAGS=$CXXFLAGS",...
    "-std=c++17", ... % To enable newer features
    "-Xclang", ...
    "-fopenmp", "-I/opt/homebrew/opt/libomp/include", ... % include OpenMP
    "-fopenmp-simd", "-fno-split-stack", ... % faster OpenMP
    "-O3", ... % optimize aggressively
    "-march=native", "-funroll-loops", "-ftree-vectorize", ... % SIMD
    "-ffast-math", ... % faster floats
    "-flto", ... % Link-Time Optimization
    ];

ldFlags = [
    "LDFLAGS=$LDFLAGS", ... 
    "-L/opt/homebrew/opt/libomp/lib", "-lomp",... % include OpenMP
    "-flto", ... % Link-Time Optimization
    ];

mex(inputFile, ...
    '-v',...
    '-outdir', outDir, ...
    strjoin(cxxFlags),...
    strjoin(ldFlags) ...
    );
