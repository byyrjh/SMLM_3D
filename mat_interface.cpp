#define NOMINMAX
#include <iostream>
#include <tiffio.h>
#pragma comment(lib, "legacy_stdio_definitions.lib")    
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <stdint.h>
#include <mat.h>
#include <matrix.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include "para_config.h"
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>
#include <math.h>
#include "miscellaneous.h"
#include "shared_struc.h"

using std::cout; using std::cin; using std::endl; using std::string; using std::vector;
using std::filesystem::current_path; using std::to_string;
extern "C"
void cuda_fitting(dim3 dimgrid, dim3 dimblock, fitting_config* para_config, const float* coef_det_d, const float* coef_exc_d, const float* data_d, const float* offset_map_d, const float* var_map_d,
	const float* gain_map_d, const float* map_ptr_x_d, const float* map_ptr_y_d, float* fitting_para_d, float* CRLBs_d, float* LogLikelihood_d, float* device_debug_d);
void cudasafe(cudaError_t err, char* str, int lineNumber);
void cudasafe(cudaError_t err, char* str, int lineNumber)
{
	if (err != cudaSuccess)
	{
		//reset all cuda devices
		int deviceCount = 0;
		int ii = 0;
		cudasafe(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", __LINE__); //query number of GPUs
		for (ii = 0; ii < deviceCount; ii++) {
			cudaSetDevice(ii);
			cudaDeviceReset();
		}
		//printf("GPUmleFit_LM:cudaFail", "%s failed with error code %i at line %d\n", str, err, lineNumber);
		exit(1); // might not stop matlab
	}
}


int main()
{
	////////////////////////////// read in fitting information //////////////////////////////////
	string Cur_dir = "M:\\Hao\\2022\\2022_12_6\\drifting check\\prt3";
	string scan_mode = "m1";
	MATFile* curent_mat;
	mxArray* pa;
	const char* name;
	string cali_path = Cur_dir + "\\setup_calibration\\";
	string seg_data_path = Cur_dir + "\\segment_data\\";
	string fitting_info = seg_data_path + "fitting_info_" + scan_mode + ".mat";
	const char* fitting_info_char = fitting_info.c_str();
	float FM_num_ptr;
	float SM_num_ptr;
	float slice_num_FM_ptr;
	float slice_num_SM_ptr;
	float num_vol_ptr;
	float cam_map_size_ptr;// row
	float cam_map_size_y_ptr;// column
	curent_mat = matOpen(fitting_info_char, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(&cam_map_size_ptr, (float*)mxGetData(pa), sizeof(float));
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(&cam_map_size_y_ptr, (float*)mxGetData(pa), sizeof(float));
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(&FM_num_ptr, (float*)mxGetData(pa), sizeof(float));
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(&SM_num_ptr, (float*)mxGetData(pa), sizeof(float));
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(&num_vol_ptr, (float*)mxGetData(pa), sizeof(float));
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(&slice_num_FM_ptr, (float*)mxGetData(pa), sizeof(float));
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(&slice_num_SM_ptr, (float*)mxGetData(pa), sizeof(float));
	
	int FM_num = (int)FM_num_ptr;
	int SM_num = (int)SM_num_ptr;
	int slice_num_FM = (int)slice_num_FM_ptr;
	int slice_num_SM = (int)slice_num_SM_ptr;
	int num_vol = (int)num_vol_ptr;
	int cuda_seg_size = 4000;  // parallel capability, limited by device
	int smooth_seg = 30;
	int smooth_seg_SM = 60; // unit sampling point
	float xybinsize_SM = 1000;  // unit nm
	int cam_map_size = (int)cam_map_size_ptr;// row
	int cam_map_size_y = (int)cam_map_size_y_ptr;// column

	//////////////////////////// read in data, PSF model and camera maps ////////////////////////
	int FM_trace = FM_num / num_vol;
	int num_seg_SM = (int)ceil((float)SM_num / (float)cuda_seg_size);
	int num_seg_FM = (int)ceil((float)FM_num / (float)cuda_seg_size);
	float* coef_det_h1 = new float[spline_x * spline_y * spline_z * num_coef_per_pix];
	float* coef_exc_h1 = new float[spline_z * num_coef_per_pix_axial];
	float* coef_det_h2 = new float[spline_x * spline_y * spline_z * num_coef_per_pix];
	float* coef_exc_h2 = new float[spline_z * num_coef_per_pix_axial];
	float* data_h_FM = new float[seg_size * seg_size * slice_num_FM * FM_num];
	float* data_h_SM = new float[seg_size * seg_size * slice_num_SM * SM_num];
	float* offset_map_h1 = new float[cam_map_size * cam_map_size_y];
	float* var_map_h1 = new float[cam_map_size * cam_map_size_y];
	float* gain_map_h1 = new float[cam_map_size * cam_map_size_y];
	float* offset_map_h2 = new float[cam_map_size * cam_map_size_y];
	float* var_map_h2 = new float[cam_map_size * cam_map_size_y];
	float* gain_map_h2 = new float[cam_map_size * cam_map_size_y];
	float* map_ptr_x_h_FM = new float[FM_num];
	float* map_ptr_y_h_FM = new float[FM_num];
	float* map_ptr_x_h_SM = new float[SM_num];
	float* map_ptr_y_h_SM = new float[SM_num];
	float* map_ptr_t_h_SM = new float[SM_num];
	
	string cam_cali_path = cali_path + "camera1_cali.mat";
	const char* cali_data_name1 = cam_cali_path.c_str();
	curent_mat = matOpen(cali_data_name1, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(gain_map_h1, (float*)mxGetData(pa), cam_map_size * cam_map_size_y * sizeof(float));
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(offset_map_h1, (float*)mxGetData(pa), cam_map_size * cam_map_size_y * sizeof(float));
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(var_map_h1, (float*)mxGetData(pa), cam_map_size * cam_map_size_y * sizeof(float));
	cam_cali_path = cali_path + "camera2_cali.mat";
	const char* cali_data_name2 = cam_cali_path.c_str();
	curent_mat = matOpen(cali_data_name2, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(gain_map_h2, (float*)mxGetData(pa), cam_map_size * cam_map_size_y * sizeof(float));
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(offset_map_h2, (float*)mxGetData(pa), cam_map_size * cam_map_size_y * sizeof(float));
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(var_map_h2, (float*)mxGetData(pa), cam_map_size * cam_map_size_y * sizeof(float));

	string coef_mx_name_det_str = cali_path + "coeff_det_exper1.mat";
	string coef_mx_name_exc_str = cali_path + "coeff_exc_exper1.mat";
	string data_FM_mx_name_str = seg_data_path + "seg_data_FM_" + scan_mode + ".mat";
	string data_SM_mx_name_str = seg_data_path + "seg_data_SM_"+ scan_mode + ".mat";
	string map_ptr_x_FM_name_str = seg_data_path + "map_ptr_x_FM_" + scan_mode + ".mat";
	string map_ptr_x_SM_name_str = seg_data_path + "map_ptr_x_SM_" + scan_mode + ".mat";
	string map_ptr_y_FM_name_str = seg_data_path + "map_ptr_y_FM_" + scan_mode + ".mat";
	string map_ptr_y_SM_name_str = seg_data_path + "map_ptr_y_SM_" + scan_mode + ".mat";
	string map_ptr_t_SM_name_str = seg_data_path + "map_ptr_t_SM_" + scan_mode + ".mat";
	
	const char* coef_mx_name_det1 = coef_mx_name_det_str.c_str();
	const char* coef_mx_name_exc1 = coef_mx_name_exc_str.c_str();
	const char* data_FM_mx_name = data_FM_mx_name_str.c_str();
	const char* map_ptr_x_FM_name = map_ptr_x_FM_name_str.c_str();
	const char* map_ptr_y_FM_name = map_ptr_y_FM_name_str.c_str();
	const char* data_SM_mx_name = data_SM_mx_name_str.c_str();
	const char* map_ptr_x_SM_name = map_ptr_x_SM_name_str.c_str();
	const char* map_ptr_y_SM_name = map_ptr_y_SM_name_str.c_str();
	const char* map_ptr_t_SM_name = map_ptr_t_SM_name_str.c_str();

	curent_mat = matOpen(coef_mx_name_det1, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(coef_det_h1, (float*)mxGetData(pa), spline_x * spline_y * spline_z * num_coef_per_pix * sizeof(float));

	curent_mat = matOpen(coef_mx_name_exc1, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(coef_exc_h1, (float*)mxGetData(pa), spline_z * num_coef_per_pix_axial * sizeof(float));
	
	coef_mx_name_det_str = cali_path + "coeff_det_exper2.mat";
	coef_mx_name_exc_str = cali_path + "coeff_exc_exper2.mat";
	const char* coef_mx_name_det2 = coef_mx_name_det_str.c_str();
	const char* coef_mx_name_exc2 = coef_mx_name_exc_str.c_str();
	curent_mat = matOpen(coef_mx_name_det2, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(coef_det_h2, (float*)mxGetData(pa), spline_x * spline_y * spline_z * num_coef_per_pix * sizeof(float));
	curent_mat = matOpen(coef_mx_name_exc2, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(coef_exc_h2, (float*)mxGetData(pa), spline_z * num_coef_per_pix_axial * sizeof(float));

	curent_mat = matOpen(data_FM_mx_name, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(data_h_FM, (float*)mxGetData(pa), seg_size * seg_size * slice_num_FM * FM_num * sizeof(float));
	
	curent_mat = matOpen(data_SM_mx_name, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(data_h_SM, (float*)mxGetData(pa), seg_size * seg_size * slice_num_SM * SM_num * sizeof(float));

	curent_mat = matOpen(map_ptr_x_FM_name, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(map_ptr_x_h_FM, (float*)mxGetData(pa), FM_num * sizeof(float));

	curent_mat = matOpen(map_ptr_y_FM_name, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(map_ptr_y_h_FM, (float*)mxGetData(pa), FM_num * sizeof(float));

	curent_mat = matOpen(map_ptr_x_SM_name, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(map_ptr_x_h_SM, (float*)mxGetData(pa), SM_num * sizeof(float));

	curent_mat = matOpen(map_ptr_y_SM_name, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(map_ptr_y_h_SM, (float*)mxGetData(pa), SM_num * sizeof(float));

	curent_mat = matOpen(map_ptr_t_SM_name, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(map_ptr_t_h_SM, (float*)mxGetData(pa), SM_num * sizeof(float));

	///////////////////////////////// initializing host variables //////////////////////////////////

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
	float* fitting_para_h = new float[FM_num * fit_para_num];
	float* fitting_para_SM_h = new float[SM_num * fit_para_num];
	float* CRLBs_h = new float[FM_num * fit_para_num];
	float* CRLBs_SM_h = new float[SM_num * fit_para_num];
	float* LogLikelihood_h = new float[FM_num];
	float* LogLikelihood_SM_h = new float[SM_num];
	float* device_debug_h = new float[FM_num * iterations * 2];
	float* device_debug_SM_h = new float[SM_num * iterations * 2];
	memset(fitting_para_h, 0, FM_num* fit_para_num * sizeof(float));
	memset(CRLBs_h, 0, FM_num* fit_para_num * sizeof(float));
	memset(LogLikelihood_h, 0, FM_num * sizeof(float));
	memset(device_debug_h, 0, FM_num* iterations*2 * sizeof(float));
	memset(fitting_para_SM_h, 0, SM_num* fit_para_num * sizeof(float));
	memset(CRLBs_SM_h, 0, SM_num* fit_para_num * sizeof(float));
	memset(LogLikelihood_SM_h, 0, SM_num * sizeof(float));
	memset(device_debug_SM_h, 0, SM_num* iterations * 2 * sizeof(float));

	int* para_config_h = new int[3];      // num_para(5 or 6)     number of emitters to fit  slice_number
	int* para_config_d;
	int deviceCount = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount(&deviceCount);
	cudaGetDeviceProperties(&deviceProp, 0);
	cudaSetDevice(0);
	//int dev_idx;
	//int use_dev = 0;
	//int num_dev = 1;
	//cudaSetValidDevices(&use_dev, num_dev);
	const size_t availableMemory = deviceProp.totalGlobalMem / 1024 / 1024;//unit MByte
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	dim3 dimBlock = block_size;  //256 threads per block   index from 0 to 255
	dim3 dimGrid;
	cudaError_t err;
	size_t free_byte;
	size_t total_byte;
	float used_mem;
	MATFile* pmat;
	mxArray* pa1;
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	/*
	double* fitting_para_h_doulbe = new double[SM_num * fit_para_num];
	string SM_os_test = seg_data_path + "fitting_result_SM_m0.mat";
	const char* SM_os_test_char = SM_os_test.c_str();
	curent_mat = matOpen(SM_os_test_char, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(fitting_para_h_doulbe, (double*)mxGetData(pa), SM_num* fit_para_num * sizeof(double));
	for (int i = 0; i < SM_num * fit_para_num; i++)
	{
		*(fitting_para_SM_h + i) = (float)*(fitting_para_h_doulbe + i);
	}

	vector<vector<vector<float>>> mat_os_SM_smo = LS_os_calc_SM(fitting_para_SM_h, map_ptr_t_h_SM, map_ptr_x_h_SM, map_ptr_y_h_SM, cam_map_size_ptr, cam_map_size_y_ptr, num_vol_ptr, SM_num, xybinsize_SM, smooth_seg_SM);
	
	float x_range = ceil(cam_map_size_ptr / (xybinsize_SM / pixel_size_cam));
	float y_range = ceil(cam_map_size_y_ptr / (xybinsize_SM / pixel_size_cam));
	float t_range = num_vol_ptr;
	double* for_matlab = new double[x_range * y_range * t_range];
	int idx_mat;
	for (int t = 0; t < t_range; t++)
	{
		for (int y = 0; y < y_range; y++)
		{
			for (int x = 0; x < x_range; x++)
			{
				idx_mat = x_range * y_range * t + y_range * y + x;
				*(for_matlab + idx_mat) = (double) mat_os_SM_smo[t][y][x];
			}
		}
	}
	string SM_os_aver = seg_data_path + "SM_os_aver.mat";
	const char* SM_os_aver_char = SM_os_aver.c_str();
	pmat = matOpen(SM_os_aver_char, "w");
	pa1 = mxCreateDoubleMatrix(x_range * y_range , t_range, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)for_matlab, x_range* y_range* t_range * sizeof(double));
	matPutVariable(pmat, "SM_os_aver", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);
	*/
	/////////////////////////////////////////////////////////////////////////////////////////////

	// fiducial marker fitting
	for (int fitting_stage = 0; fitting_stage < 2; fitting_stage++)
	{
		for (int seg_idx = 0; seg_idx < num_seg_FM; seg_idx++)
		{
			int cur_seg_size;
			int cur_ini_idx = seg_idx * cuda_seg_size;
			if (seg_idx < num_seg_FM - 1)
				cur_seg_size = cuda_seg_size;
			else
				cur_seg_size = FM_num - seg_idx * cuda_seg_size;
			// global const in
			fitting_config FM_fit_para_h(fit_para_num, cur_seg_size, slice_num_FM, cam_map_size, cam_map_size_y, 0);
			fitting_config* FM_fit_para_d;
			cudaMalloc((void**)&coef_det_d, spline_x * spline_y * spline_z * num_coef_per_pix * sizeof(float));
			cudaMalloc((void**)&coef_exc_d, spline_z * num_coef_per_pix_axial * sizeof(float));
			cudaMalloc((void**)&offset_map_d, cam_map_size * cam_map_size_y * sizeof(float));
			cudaMalloc((void**)&var_map_d, cam_map_size * cam_map_size_y * sizeof(float));
			cudaMalloc((void**)&gain_map_d, cam_map_size * cam_map_size_y * sizeof(float));
			cudaMemcpy(coef_det_d, coef_det_h2, spline_x * spline_y * spline_z * num_coef_per_pix * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(coef_exc_d, coef_exc_h2, spline_z * num_coef_per_pix_axial * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(offset_map_d, offset_map_h2, cam_map_size * cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(var_map_d, var_map_h2, cam_map_size * cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(gain_map_d, gain_map_h2, cam_map_size * cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice);
			// global argument in
			cudaMalloc((void**)&data_d, seg_size * seg_size * slice_num_FM * cur_seg_size * sizeof(float));
			cudaMalloc((void**)&map_ptr_x_d, cur_seg_size * sizeof(float));
			cudaMalloc((void**)&map_ptr_y_d, cur_seg_size * sizeof(float));
			//cudaMalloc((void**)&para_config_d, 3 * sizeof(int));
			cudaMemset(data_d, 0, seg_size * seg_size * slice_num_FM * cur_seg_size * sizeof(float));
			cudaMemset(map_ptr_x_d, 0, cur_seg_size * sizeof(float));
			cudaMemset(map_ptr_y_d, 0, cur_seg_size * sizeof(float));
			//cudaMemset(para_config_d, 0, 3 * sizeof(int));
			// global argument out
			cudaMalloc((void**)&fitting_para_d, fit_para_num * cur_seg_size * sizeof(float));   // if I can only allocate memory without initialization???
			cudaMalloc((void**)&CRLBs_d, fit_para_num * cur_seg_size * sizeof(float));
			cudaMalloc((void**)&LogLikelihood_d, cur_seg_size * sizeof(float));
			cudaMalloc((void**)&device_debug_d, cur_seg_size * iterations * 2 * sizeof(float));
			cudaMemset(fitting_para_d, 0, fit_para_num * cur_seg_size * sizeof(float));
			cudaMemset(CRLBs_d, 0, fit_para_num * cur_seg_size * sizeof(float));
			cudaMemset(LogLikelihood_d, 0, cur_seg_size * sizeof(float));
			cudaMemset(device_debug_d, 0, cur_seg_size * iterations * 2 * sizeof(float));
			dimGrid = ceil((float)cur_seg_size / (float)block_size);
			if (fitting_stage == 0)
				FM_fit_para_h.num_fitting_para = fit_para_num;
			else
				FM_fit_para_h.num_fitting_para = (fit_para_num - 1);
			//*(para_config_h + 1) = cur_seg_size;
			//*(para_config_h + 2) = slice_num_FM;
			cudaMalloc((void**)&FM_fit_para_d, sizeof(FM_fit_para_h));
			cudaMemcpy(FM_fit_para_d, &FM_fit_para_h, sizeof(FM_fit_para_h), cudaMemcpyHostToDevice);
			cudaMemcpy(data_d, data_h_FM + cur_ini_idx * seg_size * seg_size * slice_num_FM, seg_size * seg_size * slice_num_FM * cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(map_ptr_x_d, map_ptr_x_h_FM + cur_ini_idx, cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(map_ptr_y_d, map_ptr_y_h_FM + cur_ini_idx, cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			//cudaMemcpy(para_config_d, para_config_h, 3 * sizeof(int), cudaMemcpyHostToDevice);    // LS offset estimate and initialize fitting parameter
			cudaMemcpy(fitting_para_d, fitting_para_h + cur_ini_idx * fit_para_num, fit_para_num * cur_seg_size * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(CRLBs_d, CRLBs_h + cur_ini_idx * fit_para_num, fit_para_num * cur_seg_size * sizeof(int), cudaMemcpyHostToDevice);
			cudaSetDevice(0);
			cuda_fitting(dimGrid, dimBlock, FM_fit_para_d, coef_det_d, coef_exc_d, data_d, offset_map_d, var_map_d, gain_map_d, map_ptr_x_d, map_ptr_y_d, fitting_para_d, CRLBs_d, LogLikelihood_d, device_debug_d);
			//cudaGetDevice(&dev_idx);
			//printf("current used device is %d\n", dev_idx);
			err = cudaDeviceSynchronize();
			printf("cudaDeviceSynchronize error status: %s\n", cudaGetErrorString(err));
			cudaMemGetInfo(&free_byte, &total_byte);
			used_mem = ((float)total_byte - (float)free_byte) / 1024 / 1024;
			printf("used memory is %f MB\n", used_mem);
			cudaMemcpy(fitting_para_h + cur_ini_idx * fit_para_num, fitting_para_d, fit_para_num * cur_seg_size * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(CRLBs_h + cur_ini_idx * fit_para_num, CRLBs_d, fit_para_num * cur_seg_size * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(LogLikelihood_h + cur_ini_idx, LogLikelihood_d, cur_seg_size * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(device_debug_h + cur_ini_idx * iterations * 2, device_debug_d, cur_seg_size * iterations * 2 * sizeof(float), cudaMemcpyDeviceToHost);
			err = cudaDeviceReset();
			printf("reset error status: %s\n", cudaGetErrorString(err));
			printf("Fiducial marker fitting stage %d finished\n", (fitting_stage + 1));
		}
		
		//  smooth LS offset
		if (fitting_stage == 0)
		{
			double* test_LS_os = new double[FM_trace * num_vol];
			memset(test_LS_os, 0, FM_trace * num_vol * sizeof(double));
			double* test_dist = new double[SM_num];
			memset(test_dist, 0, SM_num * sizeof(double));
			for (int j = 0; j < num_vol; j++)
			{
				int ini_idx = j - smooth_seg / 2;
				int end_idx = j + smooth_seg / 2;
				int smooth_seg_size;
				if (ini_idx < 0)
				{
					smooth_seg_size = 1 + smooth_seg + ini_idx;
					ini_idx = 0;
				}
				else if (end_idx > (num_vol - 1))
					smooth_seg_size = smooth_seg + num_vol - end_idx;
				else
					smooth_seg_size = 1 + smooth_seg;
				for (int FM_idx = 0; FM_idx < FM_trace; FM_idx++)  // FM data structure 11111 22222 33333 44444 55555 .... 12345 represent time(volume indices) 5 FMs are sorted in the same way for each block
				{
					float LS_os = 0;
					float os_counter = 0;
					int cur_idx;
					int FM_pos = j * FM_trace + FM_idx;
					for (int cur_smooth_seg = 0; cur_smooth_seg < smooth_seg_size; cur_smooth_seg++)
					{
						cur_idx = ini_idx * FM_trace + FM_idx + cur_smooth_seg * FM_trace;
						if (!isnan(*(fitting_para_h + cur_idx * fit_para_num + 5)))
						{
							LS_os += *(fitting_para_h + cur_idx * fit_para_num + 5);
							++os_counter;
						}
					}
					*(fitting_para_h + FM_pos * fit_para_num + 5) = LS_os / os_counter;
					*(test_LS_os + j * FM_trace + FM_idx) = (double)*(fitting_para_h + FM_pos * fit_para_num + 5);
				}
			}
			
			string file_test_os = seg_data_path + "LS_os_" + scan_mode + ".mat";
			const char* file_os = file_test_os.c_str();
			pmat = matOpen(file_os, "w");
			pa1 = mxCreateDoubleMatrix(FM_trace, num_vol, mxREAL);
			memcpy((void*)(mxGetPr(pa1)), (void*)test_LS_os, FM_trace* num_vol * sizeof(double));
			matPutVariable(pmat, "LS_os", pa1);
			mxDestroyArray(pa1);
			matClose(pmat);
		}

	}
	// FM fitting output
	double* fitting_para_crlb = new double[FM_num * fit_para_num];
	double* fitting_para_end = new double[FM_num * fit_para_num];
	double* fitting_para_ChiSq = new double[FM_num];
	double* device_debug_out = new double[iterations * 2 * FM_num];
	memset(fitting_para_crlb, 0, FM_num* fit_para_num * sizeof(double));
	memset(fitting_para_end, 0, FM_num* fit_para_num * sizeof(double));
	memset(fitting_para_ChiSq, 0, FM_num * sizeof(double));
	memset(device_debug_out, 0, iterations * 2 * FM_num * sizeof(double));
	for (int i = 0; i < FM_num; i++)
	{
		*(fitting_para_ChiSq + i) = (double)*(LogLikelihood_h + i);
		for (int j = 0; j < iterations * 2; j++)
		{
			*(device_debug_out + i * iterations * 2 + j) = (double)*(device_debug_h + i * iterations * 2 + j);
		}
	}
	for (int i = 0; i < FM_num * fit_para_num; i++)
	{
		*(fitting_para_crlb + i) = (double)*(CRLBs_h + i);
		*(fitting_para_end + i) = (double)*(fitting_para_h + i);
	}
	string file_crlb_full = seg_data_path + "crlb_FM_" + scan_mode + ".mat";
	string file_fitting_para_full = seg_data_path + "fitting_result_FM_" + scan_mode + ".mat";
	string finalChiSq_full = seg_data_path + "ChiSq_FM_" + scan_mode + ".mat";
	string device_debug_out_char_full = seg_data_path + "device_debug_out_FM_" + scan_mode + ".mat";
	const char* file_crlb = file_crlb_full.c_str();
	const char* file_fitting_para = file_fitting_para_full.c_str();
	const char* finalChiSq = finalChiSq_full.c_str();
	const char* device_debug_out_char = device_debug_out_char_full.c_str();

	pmat = matOpen(file_crlb, "w");
	pa1 = mxCreateDoubleMatrix(fit_para_num, FM_num, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)fitting_para_crlb, fit_para_num * FM_num * sizeof(double));
	matPutVariable(pmat, "crlb_results", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);

	pmat = matOpen(file_fitting_para, "w");
	pa1 = mxCreateDoubleMatrix(fit_para_num, FM_num, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)fitting_para_end, fit_para_num * FM_num * sizeof(double));
	matPutVariable(pmat, "fitting_results", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);

	pmat = matOpen(finalChiSq, "w");
	pa1 = mxCreateDoubleMatrix(FM_num, 1, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)fitting_para_ChiSq, FM_num * sizeof(double));
	matPutVariable(pmat, "ChiSq", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);

	pmat = matOpen(device_debug_out_char, "w");
	pa1 = mxCreateDoubleMatrix(iterations * 2, FM_num, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)device_debug_out, FM_num * iterations * 2 * sizeof(double));
	matPutVariable(pmat, "test", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);

	

	// single molecule fitting
	for (int fitting_stage = 0; fitting_stage < 2; fitting_stage++)
	{
		for (int seg_idx = 0; seg_idx < num_seg_SM; seg_idx++)
		{
			cudaSetDevice(0);
			cudaMalloc((void**)&coef_det_d, spline_x * spline_y * spline_z * num_coef_per_pix * sizeof(float));
			cudaMalloc((void**)&coef_exc_d, spline_z * num_coef_per_pix_axial * sizeof(float));
			cudaMalloc((void**)&offset_map_d, cam_map_size * cam_map_size_y * sizeof(float));
			cudaMalloc((void**)&var_map_d, cam_map_size * cam_map_size_y * sizeof(float));
			cudaMalloc((void**)&gain_map_d, cam_map_size * cam_map_size_y * sizeof(float));
			cudaMemcpy(coef_det_d, coef_det_h1, spline_x * spline_y * spline_z * num_coef_per_pix * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(coef_exc_d, coef_exc_h1, spline_z * num_coef_per_pix_axial * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(offset_map_d, offset_map_h1, cam_map_size * cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(var_map_d, var_map_h1, cam_map_size * cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(gain_map_d, gain_map_h1, cam_map_size * cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice);
			// global argument in
			int cur_seg_size;
			int cur_ini_idx = seg_idx * cuda_seg_size;
			if (seg_idx < num_seg_SM - 1)
				cur_seg_size = cuda_seg_size;
			else
				cur_seg_size = SM_num - seg_idx * cuda_seg_size;
			fitting_config SM_fit_para_h(fit_para_num, cur_seg_size, slice_num_SM, cam_map_size, cam_map_size_y, 1);
			fitting_config* SM_fit_para_d;
			cudaMalloc((void**)&data_d, seg_size * seg_size * slice_num_SM * cur_seg_size * sizeof(float));
			cudaMalloc((void**)&map_ptr_x_d, cur_seg_size * sizeof(float));
			cudaMalloc((void**)&map_ptr_y_d, cur_seg_size * sizeof(float));
			//cudaMalloc((void**)&para_config_d, 3 * sizeof(int));
			cudaMemset(data_d, 0, seg_size * seg_size * slice_num_SM * cur_seg_size * sizeof(float));
			cudaMemset(map_ptr_x_d, 0, cur_seg_size * sizeof(float));
			cudaMemset(map_ptr_y_d, 0, cur_seg_size * sizeof(float));
			//cudaMemset(para_config_d, 0, 3 * sizeof(int));
			if (fitting_stage == 0)
				SM_fit_para_h.num_fitting_para = fit_para_num;
			else
				SM_fit_para_h.num_fitting_para = (fit_para_num - 1);
			cudaMalloc((void**)&SM_fit_para_d, sizeof(SM_fit_para_h));
			cudaMemcpy(SM_fit_para_d, &SM_fit_para_h, sizeof(SM_fit_para_h), cudaMemcpyHostToDevice);
			// global argument out
			cudaMalloc((void**)&fitting_para_d, fit_para_num * cur_seg_size * sizeof(float));   // if I can only allocate memory without initialization???
			cudaMalloc((void**)&CRLBs_d, fit_para_num * cur_seg_size * sizeof(float));
			cudaMalloc((void**)&LogLikelihood_d, cur_seg_size * sizeof(float));
			cudaMalloc((void**)&device_debug_d, cur_seg_size * iterations * 2 * sizeof(float));
			cudaMemset(fitting_para_d, 0, fit_para_num * cur_seg_size * sizeof(float));
			cudaMemset(CRLBs_d, 0, fit_para_num * cur_seg_size * sizeof(float));
			cudaMemset(LogLikelihood_d, 0, cur_seg_size * sizeof(float));
			cudaMemset(device_debug_d, 0, cur_seg_size * iterations * 2 * sizeof(float));

			dimGrid = ceil((float)cur_seg_size / (float)block_size);
			*para_config_h = (fit_para_num - 1);
			*(para_config_h + 1) = cur_seg_size;
			*(para_config_h + 2) = slice_num_SM;

			cudaMemcpy(data_d, data_h_SM + cur_ini_idx * seg_size * seg_size * slice_num_SM, seg_size * seg_size * slice_num_SM * cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(map_ptr_x_d, map_ptr_x_h_SM + cur_ini_idx, cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(map_ptr_y_d, map_ptr_y_h_SM + cur_ini_idx, cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			//cudaMemcpy(para_config_d, para_config_h, 3 * sizeof(int), cudaMemcpyHostToDevice);    // LS offset estimate and initialize fitting parameter
			cudaMemcpy(fitting_para_d, fitting_para_SM_h + cur_ini_idx * fit_para_num, fit_para_num * cur_seg_size * sizeof(int), cudaMemcpyHostToDevice);
			cudaSetDevice(0);
			cuda_fitting(dimGrid, dimBlock, SM_fit_para_d, coef_det_d, coef_exc_d, data_d, offset_map_d, var_map_d, gain_map_d, map_ptr_x_d, map_ptr_y_d, fitting_para_d, CRLBs_d, LogLikelihood_d, device_debug_d);
			cudaSetDevice(0);
			err = cudaDeviceSynchronize();
			printf("cudaDeviceSynchronize error status: %s\n", cudaGetErrorString(err));
			cudaMemGetInfo(&free_byte, &total_byte);
			used_mem = ((float)total_byte - (float)free_byte) / 1024 / 1024;
			printf("used memory is %f MB\n", used_mem);

			cudaMemcpy(fitting_para_SM_h + cur_ini_idx * fit_para_num, fitting_para_d, fit_para_num * cur_seg_size * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(CRLBs_SM_h + cur_ini_idx * fit_para_num, CRLBs_d, fit_para_num * cur_seg_size * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(LogLikelihood_SM_h + cur_ini_idx, LogLikelihood_d, cur_seg_size * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(device_debug_SM_h + cur_ini_idx * iterations * 2, device_debug_d, cur_seg_size * iterations * 2 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaSetDevice(0);
			err = cudaDeviceReset();
			printf("reset error status: %s\n", cudaGetErrorString(err));
			printf("fitting stage %d, segment set %d fitting finished, %d segment sets left\n\n", fitting_stage + 1, seg_idx + 1, num_seg_SM - seg_idx - 1);
		}
		if (fitting_stage == 0)
		{
			vector<vector<vector<float>>> mat_os_SM_smo = LS_os_calc_SM(fitting_para_SM_h, map_ptr_t_h_SM, map_ptr_x_h_SM, map_ptr_y_h_SM, cam_map_size_ptr, cam_map_size_y_ptr, num_vol_ptr, SM_num, xybinsize_SM, smooth_seg_SM);
			float x_range = ceil(cam_map_size_ptr / (xybinsize_SM / pixel_size_cam));
			float y_range = ceil(cam_map_size_y_ptr / (xybinsize_SM / pixel_size_cam));
			float t_range = num_vol_ptr;
			double* for_matlab = new double[x_range * y_range * t_range];
			int idx_mat;
			for (int t = 0; t < t_range; t++)
			{
				for (int y = 0; y < y_range; y++)
				{
					for (int x = 0; x < x_range; x++)
					{
						idx_mat = x_range * y_range * t + y_range * y + x;
						*(for_matlab + idx_mat) = (double)mat_os_SM_smo[t][y][x];
					}
				}
			}
			string SM_os_aver = seg_data_path + "SM_os_aver_" + scan_mode + ".mat";
			const char* SM_os_aver_char = SM_os_aver.c_str();
			pmat = matOpen(SM_os_aver_char, "w");
			pa1 = mxCreateDoubleMatrix(x_range * y_range, t_range, mxREAL);
			memcpy((void*)(mxGetPr(pa1)), (void*)for_matlab, x_range * y_range * t_range * sizeof(double));
			matPutVariable(pmat, "SM_os_aver", pa1);
			mxDestroyArray(pa1);
			matClose(pmat);
		}
	}
	// cuda_kernel end
	// SM fitting output

	double* fitting_para_crlb_SM = new double[SM_num * fit_para_num];
	double* fitting_para_end_SM = new double[SM_num * fit_para_num];
	double* fitting_para_ChiSq_SM = new double[SM_num];
	double* device_debug_out_SM = new double[iterations * 2 * SM_num];
	memset(fitting_para_crlb_SM, 0, SM_num* fit_para_num * sizeof(double));
	memset(fitting_para_end_SM, 0, SM_num* fit_para_num * sizeof(double));
	memset(fitting_para_ChiSq_SM, 0, SM_num * sizeof(double));
	memset(device_debug_out_SM, 0, iterations * 2 * SM_num * sizeof(double));
	for (int i = 0; i < SM_num; i++)
	{
		*(fitting_para_ChiSq_SM + i) = (double)*(LogLikelihood_SM_h + i);
		for (int j = 0; j < iterations * 2; j++)
		{
			*(device_debug_out_SM + i * iterations * 2 + j) = (double)*(device_debug_SM_h + i * iterations * 2 + j);
		}
	}
	for (int i = 0; i < SM_num * fit_para_num; i++)
	{
		*(fitting_para_crlb_SM + i) = (double)*(CRLBs_SM_h + i);
		*(fitting_para_end_SM + i) = (double)*(fitting_para_SM_h + i);
	}

	string file_crlb_full_SM = seg_data_path + "crlb_SM_" + scan_mode + ".mat";
	string file_fitting_para_full_SM = seg_data_path + "fitting_result_SM_" + scan_mode + ".mat";
	string finalChiSq_full_SM = seg_data_path + "ChiSq_SM_" + scan_mode + ".mat";
	string device_debug_out_char_full_SM = seg_data_path + "device_debug_out_SM_" + scan_mode + ".mat";
	const char* file_crlb_SM = file_crlb_full_SM.c_str();
	const char* file_fitting_para_SM = file_fitting_para_full_SM.c_str();
	const char* finalChiSq_SM = finalChiSq_full_SM.c_str();
	const char* device_debug_out_char_SM = device_debug_out_char_full_SM.c_str();

	pmat = matOpen(file_crlb_SM, "w");
	pa1 = mxCreateDoubleMatrix(fit_para_num, SM_num, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)fitting_para_crlb_SM, fit_para_num * SM_num * sizeof(double));
	matPutVariable(pmat, "crlb_results", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);

	pmat = matOpen(file_fitting_para_SM, "w");
	pa1 = mxCreateDoubleMatrix(fit_para_num, SM_num, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)fitting_para_end_SM, fit_para_num * SM_num * sizeof(double));
	matPutVariable(pmat, "fitting_results", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);

	pmat = matOpen(finalChiSq_SM, "w");
	pa1 = mxCreateDoubleMatrix(SM_num, 1, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)fitting_para_ChiSq_SM, SM_num * sizeof(double));
	matPutVariable(pmat, "ChiSq", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);

	pmat = matOpen(device_debug_out_char_SM, "w");
	pa1 = mxCreateDoubleMatrix(iterations * 2, SM_num, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)device_debug_out_SM, SM_num * iterations * 2 * sizeof(double));
	matPutVariable(pmat, "test", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);
	
	delete[] fitting_para_crlb, fitting_para_end, fitting_para_ChiSq, device_debug_out;
	// .mat output end
	delete[] coef_det_h1, coef_exc_h1, data_h_FM, offset_map_h1, var_map_h1, gain_map_h1, map_ptr_x_h_FM, map_ptr_y_h_FM, para_config_h;
	delete[] fitting_para_h, CRLBs_h, LogLikelihood_h, device_debug_h;
	delete[] offset_map_h2, var_map_h2, gain_map_h2, coef_det_h2, coef_exc_h2;
	
    
}