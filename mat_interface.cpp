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
#include <stdexcept>

using std::cout; using std::cin; using std::endl; using std::string; using std::vector;
using std::filesystem::current_path; using std::to_string;
extern "C"
void cuda_fitting(dim3 dimgrid, dim3 dimblock, fitting_config* para_config, const float* coef_det_d, const float* coef_exc_d, const float* data_d, const float* offset_map_d, const float* var_map_d,
	const float* gain_map_d, const float* map_ptr_x_d, const float* map_ptr_y_d, float* fitting_para_d, float* CRLBs_d, float* LogLikelihood_d, float* device_debug_d);
void cuda_plane_coef(dim3 dimgrid, dim3 dimblock, ls_plane_config* ls_plane_info_d, float* LS_plane_fit_coef_d, const float* LS_os_map_d);
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
	string Cur_dir = "M:\\Hao\\2023\\4_25\\MT_sec_AB_5mM\\fov6";
	string scan_mode = "m1"; // m0 offset = 16  m1 offset = -16(wrong sign)  m0 should have been -16 and m1 should have been +16
	bool FM_fit = true;
	bool SM_fit = true;
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
	float stationary_pos_ptr;
	float vol_per_hyper_ptr;
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
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(&stationary_pos_ptr, (float*)mxGetData(pa), sizeof(float));
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(&vol_per_hyper_ptr, (float*)mxGetData(pa), sizeof(float));
	int FM_num = (int)FM_num_ptr;
	int SM_num = (int)SM_num_ptr;
	int slice_num_FM = (int)slice_num_FM_ptr;
	int slice_num_SM = (int)slice_num_SM_ptr;
	int num_vol = (int)num_vol_ptr;
	int cuda_seg_size = 4000;  // parallel capability, limited by device
	int smooth_seg = 30;
	int smooth_seg_SM = 94; // unit sampling point
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
	float* map_ptr_z_h_SM = new float[SM_num];
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
	string map_ptr_z_SM_name_str = seg_data_path + "map_ptr_z_SM_" + scan_mode + ".mat";
	string map_ptr_t_SM_name_str = seg_data_path + "map_ptr_t_SM_" + scan_mode + ".mat";
	
	const char* coef_mx_name_det1 = coef_mx_name_det_str.c_str();
	const char* coef_mx_name_exc1 = coef_mx_name_exc_str.c_str();
	const char* data_FM_mx_name = data_FM_mx_name_str.c_str();
	const char* map_ptr_x_FM_name = map_ptr_x_FM_name_str.c_str();
	const char* map_ptr_y_FM_name = map_ptr_y_FM_name_str.c_str();
	const char* data_SM_mx_name = data_SM_mx_name_str.c_str();
	const char* map_ptr_x_SM_name = map_ptr_x_SM_name_str.c_str();
	const char* map_ptr_y_SM_name = map_ptr_y_SM_name_str.c_str();
	const char* map_ptr_z_SM_name = map_ptr_z_SM_name_str.c_str();
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

	curent_mat = matOpen(map_ptr_z_SM_name, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(map_ptr_z_h_SM, (float*)mxGetData(pa), SM_num * sizeof(float));

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

	int deviceCount = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount(&deviceCount);
	cudaGetDeviceProperties(&deviceProp, 1);
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
	
	// fiducial marker fitting
	int fitting_times;
	bool LS_os_FM = true;
	if (FM_fit)
	{
		fitting_times = 1;
	for (int fitting_stage = 0; fitting_stage < fitting_times; fitting_stage++)
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
			fitting_config FM_fit_para_h(LS_os_FM, cur_seg_size, slice_num_FM, cam_map_size, cam_map_size_y, 0);
			fitting_config* FM_fit_para_d;
			cudaSetDevice(0);
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
			cudaMalloc((void**)&FM_fit_para_d, sizeof(FM_fit_para_h));
			cudaMemset(data_d, 0, seg_size * seg_size * slice_num_FM * cur_seg_size * sizeof(float));
			cudaMemset(map_ptr_x_d, 0, cur_seg_size * sizeof(float));
			cudaMemset(map_ptr_y_d, 0, cur_seg_size * sizeof(float));
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
			
			cudaMemcpy(FM_fit_para_d, &FM_fit_para_h, sizeof(FM_fit_para_h), cudaMemcpyHostToDevice);
			cudaMemcpy(data_d, data_h_FM + cur_ini_idx * seg_size * seg_size * slice_num_FM, seg_size * seg_size * slice_num_FM * cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(map_ptr_x_d, map_ptr_x_h_FM + cur_ini_idx, cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(map_ptr_y_d, map_ptr_y_h_FM + cur_ini_idx, cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			// To investigate localization precision vs with/without light sheet offset fitting code should be adapted here
			//cudaMemcpy(fitting_para_d, fitting_para_h + cur_ini_idx * fit_para_num, fit_para_num * cur_seg_size * sizeof(int), cudaMemcpyHostToDevice);
			//cudaMemcpy(CRLBs_d, CRLBs_h + cur_ini_idx * fit_para_num, fit_para_num * cur_seg_size * sizeof(int), cudaMemcpyHostToDevice);
			cudaSetDevice(0);
			cuda_fitting(dimGrid, dimBlock, FM_fit_para_d, coef_det_d, coef_exc_d, data_d, offset_map_d, var_map_d, gain_map_d, map_ptr_x_d, map_ptr_y_d, fitting_para_d, CRLBs_d, LogLikelihood_d, device_debug_d);
			//cudaGetDevice(&dev_idx);
			err = cudaDeviceSynchronize();
			printf("cudaDeviceSynchronize error status: %s\n", cudaGetErrorString(err));
			cudaMemGetInfo(&free_byte, &total_byte);
			used_mem = ((float)total_byte - (float)free_byte) / 1024 / 1024;
			printf("used memory is %f MB\n", used_mem);
			cudaMemcpy(fitting_para_h + cur_ini_idx * fit_para_num, fitting_para_d, fit_para_num * cur_seg_size * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(CRLBs_h + cur_ini_idx * fit_para_num, CRLBs_d, fit_para_num * cur_seg_size * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(LogLikelihood_h + cur_ini_idx, LogLikelihood_d, cur_seg_size * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(device_debug_h + cur_ini_idx * iterations * 2, device_debug_d, cur_seg_size * iterations * 2 * sizeof(float), cudaMemcpyDeviceToHost);
			cudaSetDevice(0);
			err = cudaDeviceReset();
			printf("reset error status: %s\n", cudaGetErrorString(err));
			printf("Fiducial marker fitting stage %d finished\n", (fitting_stage + 1));
		}
		
		//  smooth LS offset
		if ((fitting_stage == 0) && (slice_num_SM > 1))
		{
			double* test_LS_os = new double[FM_trace * num_vol];
			memset(test_LS_os, 0, FM_trace * num_vol * sizeof(double));
			LS_os_calc_FM(fitting_para_h, test_LS_os, num_vol, stationary_pos_ptr, vol_per_hyper_ptr, smooth_seg, FM_trace);
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
	}
	

	// single molecule fitting
	// number of slices per segment and if to fit light sheet offset should be carefully configured or otherwise the fitting kernel could fail
	if (SM_fit)
	{
		double* fitting_para_crlb_SM = new double[SM_num * fit_para_num];
		double* fitting_para_end_SM = new double[SM_num * fit_para_num];
		double* fitting_para_crlb_check_SM = new double[SM_num * fit_para_num];
		double* fitting_para_ChiSq_SM = new double[SM_num];
		double* device_debug_out_SM = new double[iterations * 2 * SM_num];
		fitting_times = 2;

	bool LS_os_SM;
	string fit_round;
	bool skip_first_round;
	string SM_fitting_res_round1 = seg_data_path + "fitting_result_SM_" + scan_mode + "_round1.mat";
	const char* SM_fitting_res_round1_char = SM_fitting_res_round1.c_str();
	string SM_fitting_res_crlb_round1 = seg_data_path + "fitting_result_crlb_SM_" + scan_mode + "_round1.mat";
	const char* SM_fitting_res_crlb_round1_char = SM_fitting_res_crlb_round1.c_str();
	std::filesystem::path fitting_res_path = SM_fitting_res_round1;
	skip_first_round = std::filesystem::exists(fitting_res_path);
	float x_range = ceil(cam_map_size_ptr / (xybinsize_SM / pixel_size_cam));
	float y_range = ceil(cam_map_size_y_ptr / (xybinsize_SM / pixel_size_cam));
	float t_range = num_vol_ptr;
	float* LS_os_data = new float[x_range * y_range * t_range];
	for (int fitting_stage = 0; fitting_stage < fitting_times; fitting_stage++)
	{
		if (fitting_stage == 0)
			LS_os_SM = true;
		else
			LS_os_SM = false;
		if (!(skip_first_round & (fitting_stage == 0)))
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
			fitting_config SM_fit_para_h(LS_os_SM, cur_seg_size, slice_num_SM, cam_map_size, cam_map_size_y, 1);
			fitting_config* SM_fit_para_d;
			cudaMalloc((void**)&data_d, seg_size * seg_size * slice_num_SM * cur_seg_size * sizeof(float));
			cudaMalloc((void**)&map_ptr_x_d, cur_seg_size * sizeof(float));
			cudaMalloc((void**)&map_ptr_y_d, cur_seg_size * sizeof(float));
			cudaMemset(data_d, 0, seg_size * seg_size * slice_num_SM * cur_seg_size * sizeof(float));
			cudaMemset(map_ptr_x_d, 0, cur_seg_size * sizeof(float));
			cudaMemset(map_ptr_y_d, 0, cur_seg_size * sizeof(float));
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

			cudaMemcpy(data_d, data_h_SM + cur_ini_idx * seg_size * seg_size * slice_num_SM, seg_size * seg_size * slice_num_SM * cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(map_ptr_x_d, map_ptr_x_h_SM + cur_ini_idx, cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(map_ptr_y_d, map_ptr_y_h_SM + cur_ini_idx, cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
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
			fit_round = "1";
		else
			fit_round = "2";

		memset(fitting_para_end_SM, 0, SM_num* fit_para_num * sizeof(double));
		memset(fitting_para_crlb_check_SM, 0, SM_num * fit_para_num * sizeof(double));
		for (int i = 0; i < SM_num * fit_para_num; i++)
		{
			*(fitting_para_end_SM + i) = (double)*(fitting_para_SM_h + i);
			*(fitting_para_crlb_check_SM + i) = (double)*(CRLBs_SM_h + i);
		}
		string file_fitting_para_full_SM = seg_data_path + "fitting_result_SM_" + scan_mode + "_round" + fit_round + ".mat";
		const char* file_fitting_para_SM = file_fitting_para_full_SM.c_str();
		pmat = matOpen(file_fitting_para_SM, "w");
		pa1 = mxCreateDoubleMatrix(fit_para_num, SM_num, mxREAL);
		memcpy((void*)(mxGetPr(pa1)), (void*)fitting_para_end_SM, fit_para_num* SM_num * sizeof(double));
		matPutVariable(pmat, "fitting_results", pa1);
		mxDestroyArray(pa1);
		matClose(pmat);

		string file_fitting_para_crlb_full_SM = seg_data_path + "fitting_result_crlb_SM_" + scan_mode + "_round" + fit_round + ".mat";
		const char* file_fitting_para_crlb_SM = file_fitting_para_crlb_full_SM.c_str();
		pmat = matOpen(file_fitting_para_crlb_SM, "w");
		pa1 = mxCreateDoubleMatrix(fit_para_num, SM_num, mxREAL);
		memcpy((void*)(mxGetPr(pa1)), (void*)fitting_para_crlb_check_SM, fit_para_num * SM_num * sizeof(double));
		matPutVariable(pmat, "fitting_results_crlb", pa1);
		mxDestroyArray(pa1);
		matClose(pmat);
		}
		
		if ((fitting_stage == 0) & skip_first_round)
		{
			double* fitting_res_mat = new double[SM_num * fit_para_num];
			curent_mat = matOpen(SM_fitting_res_round1_char, "r");
			matGetNextVariableInfo(curent_mat, &name);
			pa = matGetVariable(curent_mat, name);
			memcpy(fitting_res_mat, mxGetData(pa), SM_num * fit_para_num * sizeof(double));
			double* fitting_res_crlb_mat = new double[SM_num * fit_para_num];
			curent_mat = matOpen(SM_fitting_res_crlb_round1_char, "r");
			matGetNextVariableInfo(curent_mat, &name);
			pa = matGetVariable(curent_mat, name);
			memcpy(fitting_res_crlb_mat, mxGetData(pa), SM_num* fit_para_num * sizeof(double));
			for (int i = 0; i < SM_num * fit_para_num; i++)
			{
				*(fitting_para_SM_h + i) = (float)*(fitting_res_mat + i);
				*(CRLBs_SM_h + i) = (float)*(fitting_res_crlb_mat + i);
			}
					
		}


		else if(fitting_stage == 0)
		{
			
			

		}
		if (fitting_stage == 0)
		{
			int z_range = 50;
			vector<vector<vector<vector<float>>>> mat_os_SM_smo = LS_os_calc_SM_full(fitting_para_SM_h, CRLBs_SM_h, map_ptr_t_h_SM, map_ptr_x_h_SM, map_ptr_y_h_SM, map_ptr_z_h_SM, cam_map_size_ptr, cam_map_size_y_ptr, num_vol_ptr, SM_num, xybinsize_SM, smooth_seg_SM);
			vector<vector<vector<vector<float>>>> mat_os_st_smoothed = LS_os_filter_full(&mat_os_SM_smo, x_range, y_range, z_range, t_range);
			

			// fit light sheet plane and initialize light sheet offset for next fitting round

			double* LS_os_AO_map = new double[x_range * y_range * z_range * t_range];
			
			int arr_iter;
			for (int t = 0; t < t_range; t++)
			{
				for (int z = 0; z < z_range; z++)
				{
					for (int y = 0; y < y_range; y++)
					{
						for (int x = 0; x < x_range; x++)
						{
							arr_iter = x + y * y_range + z * x_range * y_range + t * x_range * y_range * z_range;
							*(LS_os_AO_map + arr_iter) = (double)mat_os_st_smoothed[t][z][y][x];
						}
					}
				}
			}
				//*(LS_os_AO_map + t) = (double)LS_os_data_filter[t];
			
			int time_idx;
			float x_idx;
			float y_idx;
			int AO_map_idx;
			for (int SM_idx = 0; SM_idx < SM_num; SM_idx++)
			{
				time_idx = (int)*(map_ptr_t_h_SM + SM_idx) - 1;
				float x_pos = *(map_ptr_x_h_SM + SM_idx) + round(*(fitting_para_SM_h + SM_idx * fit_para_num));
				float y_pos = *(map_ptr_y_h_SM + SM_idx) + round(*(fitting_para_SM_h + SM_idx * fit_para_num + 1));
				float z_pos = *(map_ptr_z_h_SM + SM_idx) + round((*(fitting_para_SM_h + SM_idx * fit_para_num + 2)) * step_size / LS_stepsize);
				int idx_x = (int)(ceil(x_pos / (xybinsize_SM / pixel_size_cam)) - 1);
				int idx_y = (int)(ceil(y_pos / (xybinsize_SM / pixel_size_cam)) - 1);
				int idx_z = (int)(z_pos - 1);
				if (idx_x >= 0 && idx_x < x_range && idx_y >= 0 && idx_y < y_range && idx_z >= 0 && idx_z < z_range)
				{
					if (!isnan(mat_os_st_smoothed[time_idx][idx_z][idx_y][idx_x]))
						*(fitting_para_SM_h + 5 + SM_idx * fit_para_num) = mat_os_st_smoothed[time_idx][idx_z][idx_y][idx_x];
				}
				//AO_map_idx = time_idx * x_range * y_range + y_idx * y_range + x_idx;
				//*(fitting_para_SM_h + 5 + SM_idx * fit_para_num) = *(LS_os_data_filter + AO_map_idx);
				//*(fitting_para_SM_h + 5 + SM_idx * fit_para_num) = (*(LS_plane_coeff_smo + time_idx * 3)) * x_idx + (*(LS_plane_coeff_smo + time_idx * 3 + 1)) * y_idx + *(LS_plane_coeff_smo + time_idx * 3 + 2);
			}
			
			string SM_os_AO_map = seg_data_path + "SM_os_AO_4D_map_" + scan_mode + ".mat";
			const char* SM_os_AO_map_char = SM_os_AO_map.c_str();
			curent_mat = matOpen(SM_os_AO_map_char, "w");
			pa = mxCreateDoubleMatrix(x_range * y_range * z_range, t_range, mxREAL);
			memcpy((void*)(mxGetPr(pa)), (void*)LS_os_AO_map, t_range * z_range * y_range * x_range * sizeof(double));
			matPutVariable(curent_mat, "LS_os_AO_4D_map", pa);
			mxDestroyArray(pa);
			matClose(curent_mat);
			
		}
		

	}
	// cuda_kernel end
	// SM fitting output

	
	memset(fitting_para_crlb_SM, 0, SM_num* fit_para_num * sizeof(double));
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
	}

	string file_crlb_full_SM = seg_data_path + "crlb_SM_" + scan_mode + ".mat";
	string finalChiSq_full_SM = seg_data_path + "ChiSq_SM_" + scan_mode + ".mat";
	string device_debug_out_char_full_SM = seg_data_path + "device_debug_out_SM_" + scan_mode + ".mat";
	const char* file_crlb_SM = file_crlb_full_SM.c_str();
	const char* finalChiSq_SM = finalChiSq_full_SM.c_str();
	const char* device_debug_out_char_SM = device_debug_out_char_full_SM.c_str();

	pmat = matOpen(file_crlb_SM, "w");
	pa1 = mxCreateDoubleMatrix(fit_para_num, SM_num, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)fitting_para_crlb_SM, fit_para_num * SM_num * sizeof(double));
	matPutVariable(pmat, "crlb_results", pa1);
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
	}

}