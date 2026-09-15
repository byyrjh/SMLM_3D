%buildMCToolbox.m
% 64-bit version, October 26, 2016, HPCR

% mex -lAxMultiClampMsg.lib MC_init.cpp
% mex -lAxMultiClampMsg.lib MC_read_pipette_os.cpp
% mex -lAxMultiClampMsg.lib MC_select.cpp
% mex -lAxMultiClampMsg.lib MC_set_pipette_os.cpp
% mex -lAxMultiClampMsg.lib MC_close.cpp
% mex Matlab_wrapper.cpp
% mex -setup C++
mexcuda GPUfit.cpp kernel.cu