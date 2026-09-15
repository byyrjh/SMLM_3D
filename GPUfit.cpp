#define NOMINMAX
#include <iostream>
#include <matrix.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "para_config.h"
#include "miscellaneous.h"
#include "shared_struc.h"
#include "mex.h" 

extern "C"
void cuda_fitting(dim3 dimgrid, dim3 dimblock, fitting_config* para_config, const float* coef_det_d, const float* coef_exc_d, const float* data_d, const float* offset_map_d, const float* var_map_d,
    const float* gain_map_d, const float* map_ptr_x_d, const float* map_ptr_y_d, float* fitting_para_d, float* CRLBs_d, float* LogLikelihood_d, float* device_debug_d);

void mexFunction(
    int nlhs, mxArray* plhs[],
    int nrhs, const mxArray* prhs[])
{
    const mxArray* fitting_info = prhs[0];
    const mxArray* fitting_data = prhs[1];

    auto getScalarField = [&](const char* name) -> double
    {
        const mxArray* fld = mxGetField(fitting_info, 0, name);
        double val;
        memcpy(&val, mxGetPr(fld), sizeof(double));  // fast scalar copy
        return val;
    };

    auto getFieldData = [&](const char* name) -> float*
    {
        const mxArray* fld = mxGetField(fitting_data, 0, name);
        mwSize n = mxGetNumberOfElements(fld);
        float* ptr = new float[n];
        memcpy(ptr, mxGetPr(fld), n * sizeof(float));
        return ptr;
    };

    int slice_num = static_cast<int>(getScalarField("slice_num"));
    int seg_num = static_cast<int>(getScalarField("seg_num"));
    int fit_offset = static_cast<int>(getScalarField("fit_offset"));
    int seg_size_xy = static_cast<int>(getScalarField("seg_size_xy"));
    int cam_map_size_x = static_cast<int>(getScalarField("cam_map_size_x"));
    int cam_map_size_y = static_cast<int>(getScalarField("cam_map_size_y"));

    float* coef_det_h = getFieldData("coef_det");
    float* coef_exc_h = getFieldData("coef_exc");
    float* seg_data_h = getFieldData("seg_data");
    float* fitting_para_h = getFieldData("fitting_para");

    float* offset_map_h = getFieldData("offset_map");
    float* var_map_h = getFieldData("var_map");
    float* gain_map_h = getFieldData("gain_map");

    float* map_ptr_x_h = getFieldData("map_ptr_x");
    float* map_ptr_y_h = getFieldData("map_ptr_y");

    //mexPrintf("slice_num is %d\n", slice_num);
    //mexPrintf("fit_offset is %d\n", fit_offset);
    //mexPrintf("seg_size_xy is %d\n", seg_size_xy);
    //mexPrintf("cam_map_size_x is %d\n", cam_map_size_x);
    //mexPrintf("seg_num is %d\n", seg_num);

    //cuda configure
    float* coef_det_d;
    float* coef_exc_d;
    float* data_d;
    float* offset_map_d;
    float* var_map_d;
    float* gain_map_d;
    float* map_ptr_x_d;
    float* map_ptr_y_d;
    float* fitting_para_d; // dimension   x y z h bg
    float* CRLBs_d;     // dimension 5*1000  x y z h bg
    float* LogLikelihood_d;  // dimension 1000  for the whole pixel
    float* device_debug_d;

    float* CRLBs_h = new float[seg_num * fit_para_num];
    float* LogLikelihood_h = new float[seg_num];
    float* device_debug_h = new float[seg_num * iterations * 2];
    memset(CRLBs_h, 0, seg_num * fit_para_num * sizeof(float));
    memset(LogLikelihood_h, 0, seg_num * sizeof(float));
    memset(device_debug_h, 0, seg_num * iterations * 2 * sizeof(float));

    int deviceCount = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&deviceCount);
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaSetDevice(0);  // 0 is more powerful GPU
    const size_t availableMemory = deviceProp.totalGlobalMem / 1024 / 1024;//unit MByte
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    dim3 dimBlock = block_size;  //256 threads per block   index from 0 to 255
    dim3 dimGrid;
    cudaError_t err;
    size_t free_byte;
    size_t total_byte;
    bool LS_os_flag;
    if (fit_offset == 0)
        LS_os_flag = false;
    else
        LS_os_flag = true;
    fitting_config fit_para_h(LS_os_flag, seg_num, slice_num, cam_map_size_x, cam_map_size_y, 0);
    fitting_config* fit_para_d;

    // global microscope configure in
    cudaMalloc((void**)&coef_det_d, spline_x * spline_y * spline_z * num_coef_per_pix * sizeof(float));
    cudaMalloc((void**)&coef_exc_d, spline_z * num_coef_per_pix_axial * sizeof(float));
    cudaMalloc((void**)&offset_map_d, cam_map_size_x * cam_map_size_y * sizeof(float));
    cudaMalloc((void**)&var_map_d, cam_map_size_x * cam_map_size_y * sizeof(float));
    cudaMalloc((void**)&gain_map_d, cam_map_size_x * cam_map_size_y * sizeof(float));
    cudaMemcpy(coef_det_d, coef_det_h, spline_x * spline_y * spline_z * num_coef_per_pix * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(coef_exc_d, coef_exc_h, spline_z * num_coef_per_pix_axial * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(offset_map_d, offset_map_h, cam_map_size_x * cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(var_map_d, var_map_h, cam_map_size_x * cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gain_map_d, gain_map_h, cam_map_size_x * cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice);
    // global argument in
    cudaMalloc((void**)&data_d, seg_size_xy * seg_size_xy * slice_num * seg_num * sizeof(float));
    cudaMalloc((void**)&map_ptr_x_d, seg_num * sizeof(float));
    cudaMalloc((void**)&map_ptr_y_d, seg_num * sizeof(float));
    cudaMalloc((void**)&fit_para_d, sizeof(fit_para_h));
    cudaMemcpy(fit_para_d, &fit_para_h, sizeof(fit_para_h), cudaMemcpyHostToDevice);
    cudaMemcpy(data_d, seg_data_h, seg_size_xy* seg_size_xy* slice_num* seg_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(map_ptr_x_d, map_ptr_x_h, seg_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(map_ptr_y_d, map_ptr_y_h, seg_num * sizeof(float), cudaMemcpyHostToDevice);
    // global argument out
    cudaMalloc((void**)&fitting_para_d, fit_para_num * seg_num * sizeof(float));   // if I can only allocate memory without initialization???
    cudaMalloc((void**)&CRLBs_d, fit_para_num * seg_num * sizeof(float));
    cudaMalloc((void**)&LogLikelihood_d, seg_num * sizeof(float));
    cudaMalloc((void**)&device_debug_d, seg_num * iterations * 2 * sizeof(float));
    cudaMemset(fitting_para_d, 0, fit_para_num * seg_num * sizeof(float));
    cudaMemset(CRLBs_d, 0, fit_para_num * seg_num * sizeof(float));
    cudaMemset(LogLikelihood_d, 0, seg_num * sizeof(float));
    cudaMemset(device_debug_d, 0, seg_num* iterations * 2 * sizeof(float));
    dimGrid = ceil((float)seg_num / (float)block_size);
    
    cudaMemcpy(fitting_para_d, fitting_para_h, fit_para_num * seg_num * sizeof(int), cudaMemcpyHostToDevice);
    cuda_fitting(dimGrid, dimBlock, fit_para_d, coef_det_d, coef_exc_d, data_d, offset_map_d, var_map_d, gain_map_d, map_ptr_x_d, map_ptr_y_d, fitting_para_d, CRLBs_d, LogLikelihood_d, device_debug_d);
    err = cudaDeviceSynchronize();
    cudaMemcpy(fitting_para_h, fitting_para_d, fit_para_num * seg_num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(CRLBs_h, CRLBs_d, fit_para_num * seg_num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(LogLikelihood_h, LogLikelihood_d, seg_num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(device_debug_h, device_debug_d, seg_num * iterations * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaDeviceReset();
    
    auto addField = [&](const char* name, float* src, mwSize n) 
    {
        mwSize dims[2] = { n, 1 };
        mxArray* arr = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
        memcpy(mxGetPr(arr), src, n * sizeof(float));
        mxSetField(plhs[0], 0, name, arr);
    };

    const char* fields[] = { "fitting_para", "CRLBs", "LogLikelihood", "device_debug" };
    plhs[0] = mxCreateStructMatrix(1, 1, 4, fields);
   
    addField("fitting_para", fitting_para_h, fit_para_num* seg_num);
    addField("CRLBs", CRLBs_h, fit_para_num* seg_num);
    addField("LogLikelihood", LogLikelihood_h, seg_num);
    addField("device_debug", device_debug_h, seg_num* iterations * 2);

    delete[] coef_det_h;
    delete[] coef_exc_h;
    delete[] seg_data_h;
    delete[] offset_map_h;
    delete[] var_map_h;
    delete[] gain_map_h;
    delete[] map_ptr_x_h;
    delete[] map_ptr_y_h;
    delete[] fitting_para_h;
    delete[] CRLBs_h;
    delete[] LogLikelihood_h;
    delete[] device_debug_h;
}