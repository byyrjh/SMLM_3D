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


using std::cout; using std::cin; using std::endl; using std::string; using std::vector;
using std::filesystem::current_path; using std::to_string;
extern "C"
void cuda_fitting(dim3 dimgrid, dim3 dimblock, const int* para_config, const float* coef_det_d, const float* coef_exc_d, const float* data_d, const float* offset_map_d, const float* var_map_d,
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
	/////////////////////////////////////////////////////////////////////////////////////////
	string Cur_dir = "M:\\Hao\\2022_10_24\\fixed_Hela_tubulin_Alexa\\5nM\\cell2";
	int FM_num = 20000;
	int SM_num = 666610;
	int slice_num_FM = 5;
	int slice_num_SM = 3;
	int num_vol = 4000;
	int Stack_seg_size = 50;   // within which light sheet is considered to be stable
	int cuda_seg_size = 4000;  // parallel capability, limited by device
	int smooth_seg = 30;
	////////////////////////////////////////////////////////////////////////////////////////////
	int FM_trace = FM_num / num_vol;
	int FM_seg_num = num_vol / Stack_seg_size; // has to be the integer factor of FM_num
	int num_seg_SM = (int)ceil((float)SM_num / (float)cuda_seg_size);
	int num_seg_FM = (int)ceil((float)FM_num / (float)cuda_seg_size);
	string cali_path = Cur_dir + "\\setup_calibration\\";
	string seg_data_path = Cur_dir + "\\segment_data\\";
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
	
	MATFile* curent_mat;
	mxArray* pa;
	const char* name;

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
	string data_FM_mx_name_str = seg_data_path + "seg_data_FM_m0.mat";
	string data_SM_mx_name_str = seg_data_path + "seg_data_SM_m0.mat";
	string map_ptr_x_FM_name_str = seg_data_path + "map_ptr_x_FM_m0.mat";
	string map_ptr_x_SM_name_str = seg_data_path + "map_ptr_x_SM_m0.mat";
	string map_ptr_y_FM_name_str = seg_data_path + "map_ptr_y_FM_m0.mat";
	string map_ptr_y_SM_name_str = seg_data_path + "map_ptr_y_SM_m0.mat";
	string map_ptr_t_SM_name_str = seg_data_path + "map_ptr_t_SM_m0.mat";
	
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

	// data conversion law abcd(:,:,:,1)= 1 2  5 6  abcd(:,:,:,2)= 9  10   13 14
	//                                    3 4; 7 8                 11 12 ; 15 16

	/*
	TIFF* imgstack = TIFFOpen("test_sub.tif", "r");

	if (imgstack) {
		uint32 imagelength;
		uint32 imagewidth;
		tdata_t buf;
		uint32 row, idx, cur_dir;
		uint16_t* line_elements, elements;
		uint16_t arr[250];
		TIFFGetField(imgstack, TIFFTAG_IMAGELENGTH, &imagelength);
		TIFFGetField(imgstack, TIFFTAG_IMAGEWIDTH, &imagewidth);
		uint16_t* rowdata=new uint16_t[imagewidth* imagelength* 10];
		buf = _TIFFmalloc(TIFFScanlineSize(imgstack));
		cur_dir = 0;
		do
		{
			for (row = 0; row < imagelength; row++)
			{
				TIFFReadScanline(imgstack, buf, row);
				line_elements = static_cast<uint16_t*>(buf);
				memcpy(rowdata + row * 5+ cur_dir* imagelength* imagewidth, line_elements, imagewidth * sizeof(uint16_t));
				memcpy(arr + row * 5 + cur_dir * imagelength * imagewidth, line_elements, imagewidth * sizeof(uint16_t));
			}
			cur_dir++;
		} while (TIFFReadDirectory(imgstack));

		_TIFFfree(buf);
		TIFFClose(imgstack);
		delete[] rowdata;
	}
	*/
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
	//float* offset_tally_h = new float[FM_num];
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
	const size_t availableMemory = deviceProp.totalGlobalMem / 1024 / 1024;//unit MByte
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	dim3 dimBlock = block_size;  //256 threads per block   index from 0 to 255
	dim3 dimGrid;
	cudaError_t err;
	size_t free_byte;
	size_t total_byte;
	float used_mem;
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
			cudaMalloc((void**)&para_config_d, 3 * sizeof(int));
			cudaMemset(data_d, 0, seg_size * seg_size * slice_num_FM * cur_seg_size * sizeof(float));
			cudaMemset(map_ptr_x_d, 0, cur_seg_size * sizeof(float));
			cudaMemset(map_ptr_y_d, 0, cur_seg_size * sizeof(float));
			cudaMemset(para_config_d, 0, 3 * sizeof(int));
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
				*para_config_h = fit_para_num;
			else
				*para_config_h = (fit_para_num - 1);
			*(para_config_h + 1) = cur_seg_size;
			*(para_config_h + 2) = slice_num_FM;
			cudaMemcpy(data_d, data_h_FM + cur_ini_idx * seg_size * seg_size * slice_num_FM, seg_size * seg_size * slice_num_FM * cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(map_ptr_x_d, map_ptr_x_h_FM + cur_ini_idx, cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(map_ptr_y_d, map_ptr_y_h_FM + cur_ini_idx, cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(para_config_d, para_config_h, 3 * sizeof(int), cudaMemcpyHostToDevice);    // LS offset estimate and initialize fitting parameter
			cudaMemcpy(fitting_para_d, fitting_para_h + cur_ini_idx * fit_para_num, fit_para_num * cur_seg_size * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(CRLBs_d, CRLBs_h + cur_ini_idx * fit_para_num, fit_para_num * cur_seg_size * sizeof(int), cudaMemcpyHostToDevice);
			cuda_fitting(dimGrid, dimBlock, para_config_d, coef_det_d, coef_exc_d, data_d, offset_map_d, var_map_d, gain_map_d, map_ptr_x_d, map_ptr_y_d, fitting_para_d, CRLBs_d, LogLikelihood_d, device_debug_d);
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
		
		//  linearly fit lightsheet offset as the function of y_pos(light sheet propagating direction)
		if (fitting_stage == 0)
		{
			int cur_idx;
			float* y_mat = new float[FM_seg_num * FM_trace];
			float* os_mat = new float[FM_seg_num * FM_trace];
			float* lin_coef = new float[FM_seg_num * 2];
			for (int i = 0; i < FM_seg_num; i++) //18 segs
			{
				for (int j = 0; j < FM_trace; j++)  //6 markers
				{
					float os_count = 0;
					float pos_y = 0;
					float cur_os = 0;
					for (int k = 0; k < Stack_seg_size; k++) //50 samplings
					{
						cur_idx = Stack_seg_size * FM_trace * i + FM_trace * k + j;
						pos_y += *(map_ptr_y_h_FM + cur_idx);
						if (!isnan(*(fitting_para_h + cur_idx * fit_para_num + 5)))
						{
							cur_os += *(fitting_para_h + cur_idx * fit_para_num + 5);
							++os_count;
						}
					}
					pos_y /= Stack_seg_size;
					cur_os /= os_count;
					*(y_mat + i * FM_trace + j) = pos_y;
					*(os_mat + i * FM_trace + j) = cur_os;
				}
				float xsum = 0, x2sum = 0, ysum = 0, xysum = 0;
				for (int j = 0; j < FM_trace; j++)
				{
					xsum += *(y_mat + i * FM_trace + j);
					ysum += *(os_mat + i * FM_trace + j);
					x2sum += (float)pow(*(y_mat + i * FM_trace + j),2);
					xysum += (*(y_mat + i * FM_trace + j)) * (*(os_mat + i * FM_trace + j));
				}
				*(lin_coef + i * 2) = (FM_trace * xysum - xsum * ysum) / (FM_trace * x2sum - xsum * xsum);
				*(lin_coef + i * 2 + 1) = (x2sum * ysum - xsum * xysum) / (x2sum * FM_trace - xsum * xsum);
				// printf("a equals to % f and b equals to % f\n", *(lin_coef + i * 2), *(lin_coef + i * 2 + 1));
			}

			for (int i = 0; i < SM_num; i++)
			{
				float idx_stack = *(map_ptr_t_h_SM + i) - 1;
				int seg_idx = idx_stack / Stack_seg_size;
				float y_pos = *(map_ptr_y_h_SM + i);
				*(fitting_para_SM_h + i * fit_para_num + 5) = 0;// y_pos* (*(lin_coef + seg_idx * 2)) + *(lin_coef + seg_idx * 2 + 1);
			}

			//////////////////////////// old version //////////////////////////////
			/*
			int FM_seg_size = FM_num / FM_seg_num; //FM_seg_num=18
			vector<float> offset_mean;
			for (int i = 0; i < FM_seg_num; i++)
			{
				float offset_temp = 0;
				float counter = 0;
				for (int j = 0; j < FM_seg_size; j++)
				{
					int idx = i * FM_seg_size * fit_para_num + j * fit_para_num + 5;
					float cur_os = *(fitting_para_h + idx);
					*(offset_tally_h + FM_seg_size * i + j) = cur_os;
					if (!isnan(cur_os))
					{
						offset_temp += cur_os;
						++counter;
					}
				}
				offset_temp /= counter;
				offset_mean.push_back(offset_temp);
				for (int j = 0; j < FM_seg_size; j++)
				{
					int idx = i * FM_seg_size * fit_para_num + j * fit_para_num + 5;
					*(fitting_para_h + idx) = offset_temp;
				}
			}
			for (int i = 0; i < SM_num; i++)
			{
				float idx_stack = *(map_ptr_t_h_SM + i)-1;
				int seg_idx = idx_stack / Stack_seg_size;
				*(fitting_para_SM_h + i * fit_para_num + 5) = offset_mean[seg_idx];
			}
			*/
		}

	}
	
	// single molecule fitting
	for (int seg_idx = 0; seg_idx < num_seg_SM; seg_idx++)
	{
		cudaMalloc((void**)&coef_det_d, spline_x* spline_y* spline_z* num_coef_per_pix * sizeof(float));
		cudaMalloc((void**)&coef_exc_d, spline_z* num_coef_per_pix_axial * sizeof(float));
		cudaMalloc((void**)&offset_map_d, cam_map_size* cam_map_size_y * sizeof(float));
		cudaMalloc((void**)&var_map_d, cam_map_size* cam_map_size_y * sizeof(float));
		cudaMalloc((void**)&gain_map_d, cam_map_size* cam_map_size_y * sizeof(float));
		cudaMemcpy(coef_det_d, coef_det_h1, spline_x* spline_y* spline_z* num_coef_per_pix * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(coef_exc_d, coef_exc_h1, spline_z* num_coef_per_pix_axial * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(offset_map_d, offset_map_h1, cam_map_size* cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(var_map_d, var_map_h1, cam_map_size* cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gain_map_d, gain_map_h1, cam_map_size* cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice);
		// global argument in
		int cur_seg_size;
		int cur_ini_idx= seg_idx * cuda_seg_size;
		if (seg_idx < num_seg_SM - 1)
			cur_seg_size = cuda_seg_size;
		else
			cur_seg_size = SM_num - seg_idx * cuda_seg_size;
		cudaMalloc((void**)&data_d, seg_size* seg_size* slice_num_SM* cur_seg_size * sizeof(float));
		cudaMalloc((void**)&map_ptr_x_d, cur_seg_size * sizeof(float));
		cudaMalloc((void**)&map_ptr_y_d, cur_seg_size * sizeof(float));
		cudaMalloc((void**)&para_config_d, 3 * sizeof(int));
		cudaMemset(data_d, 0, seg_size* seg_size* slice_num_SM* cur_seg_size * sizeof(float));
		cudaMemset(map_ptr_x_d, 0, cur_seg_size * sizeof(float));
		cudaMemset(map_ptr_y_d, 0, cur_seg_size * sizeof(float));
		cudaMemset(para_config_d, 0, 3 * sizeof(int));
		// global argument out
		cudaMalloc((void**)&fitting_para_d, fit_para_num* cur_seg_size * sizeof(float));   // if I can only allocate memory without initialization???
		cudaMalloc((void**)&CRLBs_d, fit_para_num* cur_seg_size * sizeof(float));
		cudaMalloc((void**)&LogLikelihood_d, cur_seg_size * sizeof(float));
		cudaMalloc((void**)&device_debug_d, cur_seg_size* iterations * 2 * sizeof(float));
		cudaMemset(fitting_para_d, 0, fit_para_num* cur_seg_size * sizeof(float));
		cudaMemset(CRLBs_d, 0, fit_para_num* cur_seg_size * sizeof(float));
		cudaMemset(LogLikelihood_d, 0, cur_seg_size * sizeof(float));
		cudaMemset(device_debug_d, 0, cur_seg_size* iterations * 2 * sizeof(float));

		dimGrid = ceil((float)cur_seg_size / (float)block_size);
		*para_config_h = (fit_para_num - 1);
		*(para_config_h + 1) = cur_seg_size;
		*(para_config_h + 2) = slice_num_SM;
		
		cudaMemcpy(data_d, data_h_SM + cur_ini_idx * seg_size * seg_size * slice_num_SM, seg_size * seg_size * slice_num_SM * cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(map_ptr_x_d, map_ptr_x_h_SM + cur_ini_idx, cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(map_ptr_y_d, map_ptr_y_h_SM + cur_ini_idx, cur_seg_size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(para_config_d, para_config_h, 3 * sizeof(int), cudaMemcpyHostToDevice);    // LS offset estimate and initialize fitting parameter
		cudaMemcpy(fitting_para_d, fitting_para_SM_h + cur_ini_idx * fit_para_num, fit_para_num * cur_seg_size * sizeof(int), cudaMemcpyHostToDevice);

		cuda_fitting(dimGrid, dimBlock, para_config_d, coef_det_d, coef_exc_d, data_d, offset_map_d, var_map_d, gain_map_d, map_ptr_x_d, map_ptr_y_d, fitting_para_d, CRLBs_d, LogLikelihood_d, device_debug_d);

		err = cudaDeviceSynchronize();
		printf("cudaDeviceSynchronize error status: %s\n", cudaGetErrorString(err));
		cudaMemGetInfo(&free_byte, &total_byte);
		used_mem = ((float)total_byte - (float)free_byte) / 1024 / 1024;
		printf("used memory is %f MB\n", used_mem);

		cudaMemcpy(fitting_para_SM_h + cur_ini_idx * fit_para_num, fitting_para_d, fit_para_num * cur_seg_size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(CRLBs_SM_h + cur_ini_idx * fit_para_num, CRLBs_d, fit_para_num * cur_seg_size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(LogLikelihood_SM_h + cur_ini_idx, LogLikelihood_d, cur_seg_size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(device_debug_SM_h+ cur_ini_idx * iterations * 2, device_debug_d, cur_seg_size* iterations * 2 * sizeof(float), cudaMemcpyDeviceToHost);
		err = cudaDeviceReset();
		printf("reset error status: %s\n", cudaGetErrorString(err));
		printf("segment set %d fitting finished, %d segment sets left\n\n", seg_idx + 1, num_seg_SM - seg_idx - 1);
	}
	// cuda_kernel end
	// out put
		
	double* fitting_para_crlb = new double[FM_num * fit_para_num];
	double* fitting_para_end = new double[FM_num * fit_para_num];
	double* fitting_para_ChiSq = new double[FM_num];
	double* device_debug_out = new double[iterations*2 * FM_num];
	double* offset_tally_out = new double[FM_num];
	memset(fitting_para_crlb, 0, FM_num* fit_para_num * sizeof(double));
	memset(fitting_para_end, 0, FM_num* fit_para_num * sizeof(double));
	memset(fitting_para_ChiSq, 0, FM_num * sizeof(double));
	memset(device_debug_out, 0, iterations * 2 * FM_num * sizeof(double));
	memset(offset_tally_out, 0, FM_num * sizeof(double));
	for (int i = 0; i < FM_num; i++)
	{
		*(fitting_para_ChiSq + i) = (double)*(LogLikelihood_h + i);
		// *(offset_tally_out + i) = (double)*(offset_tally_h + i);
		for (int j = 0; j < iterations*2; j++)
		{
			*(device_debug_out + i * iterations*2 + j) = (double)*(device_debug_h + i * iterations*2 + j);
		}
	}
	for (int i = 0; i < FM_num * fit_para_num; i++)
	{
		*(fitting_para_crlb + i) = (double)*(CRLBs_h + i);
		*(fitting_para_end + i) = (double)*(fitting_para_h + i);
	}
	MATFile* pmat;
	mxArray* pa1;
	string file_crlb_full = seg_data_path + "crlb_FM_m0.mat";
	string file_fitting_para_full = seg_data_path + "fitting_result_FM_m0.mat";
	string finalChiSq_full = seg_data_path + "ChiSq_FM_m0.mat";
	string device_debug_out_char_full = seg_data_path + "device_debug_out_FM_m0.mat";
	//string file_offset_tally_out = seg_data_path + "offset_tally_FM_m0.mat";
	const char* file_crlb = file_crlb_full.c_str();
	const char* file_fitting_para = file_fitting_para_full.c_str();
	const char* finalChiSq = finalChiSq_full.c_str();
	const char* device_debug_out_char = device_debug_out_char_full.c_str();
	//const char* offset_tally_out_char = file_offset_tally_out.c_str();

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
	pa1 = mxCreateDoubleMatrix(iterations*2, FM_num, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)device_debug_out, FM_num* iterations*2 * sizeof(double));
	matPutVariable(pmat, "test", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);
	/*
	pmat = matOpen(offset_tally_out_char, "w");
	pa1 = mxCreateDoubleMatrix(FM_num, 1, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)offset_tally_out, FM_num * sizeof(double));
	matPutVariable(pmat, "offset_tally", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);
	*/
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

	string file_crlb_full_SM = seg_data_path + "crlb_SM_m0.mat";
	string file_fitting_para_full_SM = seg_data_path + "fitting_result_SM_m0.mat";
	string finalChiSq_full_SM = seg_data_path + "ChiSq_SM_m0.mat";
	string device_debug_out_char_full_SM = seg_data_path + "device_debug_out_SM_m0.mat";
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