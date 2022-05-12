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
	string Cur_dir = current_path().parent_path().string();
	string data_path = Cur_dir + "\\segment_data\\";
	string cali_path = Cur_dir + "\\setup_calibration\\";
	string seg_data_path = Cur_dir + "\\segment_data\\";
	float* coef_det_h = new float[spline_x * spline_y * spline_z * num_coef_per_pix];
	float* coef_exc_h = new float[spline_z * num_coef_per_pix_axial];
	float* data_h = new float[seg_size * seg_size * slice_num * emitter_num];
	float* offset_map_h = new float[cam_map_size * cam_map_size_y];
	float* var_map_h = new float[cam_map_size * cam_map_size_y];
	float* gain_map_h = new float[cam_map_size * cam_map_size_y];
	float* map_ptr_x_h = new float[emitter_num];
	float* map_ptr_y_h = new float[emitter_num];
	float* log_file_h = new float[emitter_num * 2];
	MATFile* curent_mat;
	mxArray* pa;
	const char* name;
	if (Data_type == 1)  // Experiment data
	{
		string cam_cali_path = cali_path + "camera";
		cam_cali_path += to_string(CamIdx);
		cam_cali_path += "_";
		cam_cali_path += to_string(Exposuretime);
		cam_cali_path += "ms.mat";
		const char* cali_data_name = cam_cali_path.c_str();
		curent_mat = matOpen(cali_data_name, "r");
		matGetNextVariableInfo(curent_mat, &name);
		pa = matGetVariable(curent_mat, name);
		memcpy(gain_map_h, (float*)mxGetData(pa), cam_map_size * cam_map_size_y * sizeof(float));
		matGetNextVariableInfo(curent_mat, &name);
		pa = matGetVariable(curent_mat, name);
		memcpy(offset_map_h, (float*)mxGetData(pa), cam_map_size * cam_map_size_y * sizeof(float));
		matGetNextVariableInfo(curent_mat, &name);
		pa = matGetVariable(curent_mat, name);
		memcpy(var_map_h, (float*)mxGetData(pa), cam_map_size * cam_map_size_y * sizeof(float));
	}
	else if (Data_type == 2)  // Simulation data
	{
		string offset_path_simu = cali_path + "offset_map.mat";
		string var_path_simu = cali_path + "var_map.mat";
		string gain_path_simu = cali_path + "gain_map.mat";
		const char* offset_map_name = offset_path_simu.c_str();
		const char* var_map_name = var_path_simu.c_str();
		const char* gain_map_name = gain_path_simu.c_str();
		curent_mat = matOpen(offset_map_name, "r");
		matGetNextVariableInfo(curent_mat, &name);
		pa = matGetVariable(curent_mat, name);
		memcpy(offset_map_h, (float*)mxGetData(pa), cam_map_size * cam_map_size_y * sizeof(float));
		curent_mat = matOpen(var_map_name, "r");
		matGetNextVariableInfo(curent_mat, &name);
		pa = matGetVariable(curent_mat, name);
		memcpy(var_map_h, (float*)mxGetData(pa), cam_map_size * cam_map_size_y * sizeof(float));
		curent_mat = matOpen(gain_map_name, "r");
		matGetNextVariableInfo(curent_mat, &name);
		pa = matGetVariable(curent_mat, name);
		memcpy(gain_map_h, (float*)mxGetData(pa), cam_map_size * cam_map_size_y * sizeof(float));
	}
	else
	{
		return 0;
	}
	string coef_mx_name_det_str = seg_data_path + "coeff_det_exper.mat";
	string coef_mx_name_exc_str = seg_data_path + "coeff_exc_exper.mat";
	string data_mx_name_str = seg_data_path + "seg_data.mat";
	string map_ptr_x_name_str = seg_data_path + "map_ptr_x.mat";
	string map_ptr_y_name_str = seg_data_path + "map_ptr_y.mat";
	string log_file_name_str = seg_data_path + "log_file.mat";
	const char* coef_mx_name_det = coef_mx_name_det_str.c_str();
	const char* coef_mx_name_exc = coef_mx_name_exc_str.c_str();
	const char* data_mx_name = data_mx_name_str.c_str();
	const char* map_ptr_x_name = map_ptr_x_name_str.c_str();
	const char* map_ptr_y_name = map_ptr_y_name_str.c_str();
	const char* log_file_name = log_file_name_str.c_str();

	curent_mat = matOpen(coef_mx_name_det, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(coef_det_h, (float*)mxGetData(pa), spline_x * spline_y * spline_z * num_coef_per_pix * sizeof(float));

	curent_mat = matOpen(coef_mx_name_exc, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(coef_exc_h, (float*)mxGetData(pa), spline_z * num_coef_per_pix_axial * sizeof(float));

	curent_mat = matOpen(data_mx_name, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(data_h, (float*)mxGetData(pa), seg_size * seg_size * slice_num * emitter_num * sizeof(float));

	curent_mat = matOpen(map_ptr_x_name, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(map_ptr_x_h, (float*)mxGetData(pa), emitter_num * sizeof(float));

	curent_mat = matOpen(map_ptr_y_name, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(map_ptr_y_h, (float*)mxGetData(pa), emitter_num * sizeof(float));

	curent_mat = matOpen(log_file_name, "r");
	matGetNextVariableInfo(curent_mat, &name);
	pa = matGetVariable(curent_mat, name);
	memcpy(log_file_h, (float*)mxGetData(pa), emitter_num * 2 * sizeof(float));

	vector<float> emitter_m0;
	vector<float> emitter_m1;
	for (int i = 0; i < emitter_num; i++)
	{
		if (*(log_file_h + emitter_num + i) == 0)
		{
			emitter_m0.push_back(*(log_file_h + i));
		}
		else
			emitter_m1.push_back(*(log_file_h + i));
	}
	int calc_seg = (int)ceil(emitter_m0.back() / calc_seg_size) * 2;
	vector<int> calc_seg_length(calc_seg);
	for (int i = 0; i < calc_seg / 2; i++)
	{
		int temp_end = min(calc_seg_size * (i + 1), emitter_m0.back()) + 1;
		int temp_length = 0;
		while ((emitter_m0.begin() != emitter_m0.end()) && (*(emitter_m0.begin()) < temp_end))
		{
			temp_length++;
			emitter_m0.erase(emitter_m0.begin());
		}
		calc_seg_length[i] = temp_length;
		temp_end = min(calc_seg_size * (i + 1), emitter_m1.back()) + 1;
		temp_length = 0;
		while ((emitter_m1.begin() != emitter_m1.end()) && (*(emitter_m1.begin()) < temp_end))
		{
			temp_length++;
			emitter_m1.erase(emitter_m1.begin());
		}
		calc_seg_length[i + calc_seg / 2] = temp_length;
	}

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
	float* fitting_para_h = new float[emitter_num * fit_para_num];
	float* CRLBs_h = new float[emitter_num * fit_para_num];
	float* LogLikelihood_h = new float[emitter_num];
	float* device_debug_h = new float[emitter_num * iterations*2];
	memset(fitting_para_h, 0, emitter_num * fit_para_num * sizeof(float));
	memset(CRLBs_h, 0, emitter_num * fit_para_num * sizeof(float));
	memset(LogLikelihood_h, 0, emitter_num * sizeof(float));
	memset(device_debug_h, 0, emitter_num * iterations*2 * sizeof(float));

	int* para_config_h = new int[3];      // num_para(5 or 6)   initial emitter idx    number of emitters to fit
	int* para_config_d;
	int deviceCount = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount(&deviceCount);
	cudaGetDeviceProperties(&deviceProp, 0);
	const size_t availableMemory = deviceProp.totalGlobalMem / 1024 / 1024;//unit MByte
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	// cuda_kernel start 
	dim3 dimBlock = block_size;  //256 threads per block   index from 0 to 255
	dim3 dimGrid;
	for (int iter = 0; iter < 1; iter++)//calc_seg
	{
		cudasafe(cudaMalloc((void**)&coef_det_d, spline_x * spline_y * spline_z * num_coef_per_pix * sizeof(float)), "Mem alloc for PSF_det failed.", __LINE__);
		cudasafe(cudaMalloc((void**)&coef_exc_d, spline_z * num_coef_per_pix_axial * sizeof(float)), "Mem alloc for PSF_exc failed.", __LINE__);
		cudasafe(cudaMalloc((void**)&data_d, seg_size * seg_size * slice_num * emitter_num * sizeof(float)), "Mem alloc for seg_data failed.", __LINE__);
		cudasafe(cudaMalloc((void**)&offset_map_d, cam_map_size * cam_map_size_y * sizeof(float)), "Mem alloc for offset_map failed.", __LINE__);
		cudasafe(cudaMalloc((void**)&var_map_d, cam_map_size * cam_map_size_y * sizeof(float)), "Mem alloc for var_map failed.", __LINE__);
		cudasafe(cudaMalloc((void**)&gain_map_d, cam_map_size * cam_map_size_y * sizeof(float)), "Mem alloc for gain_map failed.", __LINE__);
		cudasafe(cudaMalloc((void**)&map_ptr_x_d, emitter_num * sizeof(float)), "Mem alloc for LUT_x failed.", __LINE__);
		cudasafe(cudaMalloc((void**)&map_ptr_y_d, emitter_num * sizeof(float)), "Mem alloc for LUT_y failed.", __LINE__);
		cudasafe(cudaMalloc((void**)&fitting_para_d, fit_para_num * emitter_num * sizeof(float)), "Mem alloc for fitting_parameters failed.", __LINE__);
		cudasafe(cudaMalloc((void**)&CRLBs_d, fit_para_num * emitter_num * sizeof(float)), "Mem alloc for CRLB failed.", __LINE__);
		cudasafe(cudaMalloc((void**)&LogLikelihood_d, emitter_num * sizeof(float)), "Mem alloc for log_likelihood failed.", __LINE__);
		cudasafe(cudaMalloc((void**)&device_debug_d, emitter_num * iterations*2 * sizeof(float)), "Mem alloc for device_debug failed.", __LINE__);
		cudasafe(cudaMalloc((void**)&para_config_d, 3 * sizeof(int)), "Mem alloc for num_para failed.", __LINE__);

		cudasafe(cudaMemcpy(coef_det_d, coef_det_h, spline_x * spline_y * spline_z * num_coef_per_pix * sizeof(float), cudaMemcpyHostToDevice), "Memory for PSF_det copy failed", __LINE__);
		cudasafe(cudaMemcpy(coef_exc_d, coef_exc_h, spline_z * num_coef_per_pix_axial * sizeof(float), cudaMemcpyHostToDevice), "Memory for PSF_exc copy failed", __LINE__);
		cudasafe(cudaMemcpy(data_d, data_h, seg_size * seg_size * slice_num * emitter_num * sizeof(float), cudaMemcpyHostToDevice), "Memory for seg_data copy failed", __LINE__);
		cudasafe(cudaMemcpy(offset_map_d, offset_map_h, cam_map_size * cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice), "Memory for offset_map copy failed", __LINE__);
		cudasafe(cudaMemcpy(var_map_d, var_map_h, cam_map_size * cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice), "Memory for var_map copy failed", __LINE__);
		cudasafe(cudaMemcpy(gain_map_d, gain_map_h, cam_map_size * cam_map_size_y * sizeof(float), cudaMemcpyHostToDevice), "Memory for gain_map copy failed", __LINE__);
		cudasafe(cudaMemcpy(map_ptr_x_d, map_ptr_x_h, emitter_num * sizeof(float), cudaMemcpyHostToDevice), "Memory for LUT_x copy failed", __LINE__);
		cudasafe(cudaMemcpy(map_ptr_y_d, map_ptr_y_h, emitter_num * sizeof(float), cudaMemcpyHostToDevice), "Memory for LUT_y copy failed", __LINE__);
		cudasafe(cudaMemcpy(fitting_para_d, fitting_para_h, fit_para_num * emitter_num * sizeof(float), cudaMemcpyHostToDevice), "Memory for fitting_para copy failed.", __LINE__);
		cudasafe(cudaMemcpy(CRLBs_d, CRLBs_h, fit_para_num * emitter_num * sizeof(float), cudaMemcpyHostToDevice), "Memory for CRLBs copy failed", __LINE__);
		cudasafe(cudaMemcpy(LogLikelihood_d, LogLikelihood_h, emitter_num * sizeof(float), cudaMemcpyHostToDevice), "Memory for LogLikelihood copy failed.", __LINE__);

		int emitter_ini = 1;
		for (int j = 0; j < iter; j++)
			emitter_ini += calc_seg_length[j];
		int cur_seg_size = calc_seg_length[iter];
		*(para_config_h + 1) = emitter_ini;
		*(para_config_h + 2) = cur_seg_size;
		dimGrid = ceil((float)cur_seg_size / (float)block_size);
		*para_config_h = fit_para_num;
		cudasafe(cudaMemcpy(para_config_d, para_config_h, 3 * sizeof(int), cudaMemcpyHostToDevice), "Memory for num_para copy failed", __LINE__);    // LS offset estimate and initialize fitting parameter
		cuda_fitting(dimGrid, dimBlock, para_config_d, coef_det_d, coef_exc_d, data_d, offset_map_d, var_map_d, gain_map_d, map_ptr_x_d, map_ptr_y_d, fitting_para_d, CRLBs_d, LogLikelihood_d, device_debug_d);
		cudaError_t err = cudaGetLastError();
		printf("error status: %s\n", cudaGetErrorString(err));
		err=cudaDeviceSynchronize();
		printf("error status: %s\n", cudaGetErrorString(err));
		*para_config_h = fit_para_num - 1;
		cudasafe(cudaMemcpy(para_config_d, para_config_h, 3 * sizeof(int), cudaMemcpyHostToDevice), "Memory for num_para copy failed", __LINE__);
		cudasafe(cudaMemcpy(device_debug_d, device_debug_h, emitter_num * iterations*2 * sizeof(float), cudaMemcpyHostToDevice), "Memory for num_para device_debug failed.", __LINE__);   // fine localization
		cuda_fitting(dimGrid, dimBlock, para_config_d, coef_det_d, coef_exc_d, data_d, offset_map_d, var_map_d, gain_map_d, map_ptr_x_d, map_ptr_y_d, fitting_para_d, CRLBs_d, LogLikelihood_d, device_debug_d);
		err=cudaDeviceSynchronize();
		printf("error status: %s\n", cudaGetErrorString(err));
		cudasafe(cudaMemcpy(fitting_para_h, fitting_para_d, fit_para_num * emitter_num * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy failed for fitting_parameters.", __LINE__);
		cudasafe(cudaMemcpy(CRLBs_h, CRLBs_d, fit_para_num * emitter_num * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy failed for CRLB.", __LINE__);
		cudasafe(cudaMemcpy(LogLikelihood_h, LogLikelihood_d, emitter_num * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy failed for log_likelihood.", __LINE__);
		cudasafe(cudaMemcpy(device_debug_h, device_debug_d, emitter_num * iterations*2 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy failed for device_debug.", __LINE__);
		cudasafe(cudaDeviceReset(), "sync failed", __LINE__);
		printf("segment set %d fitting finished, %d segment sets left\n", iter+1, calc_seg-iter-1);
	}
	// cuda_kernel end

	/*
	for (int i = 0; i < 10; i++)
	{
		printf("x_ini=%f,x_end=%f, y_ini=%f,y_end=%f, z_ini=%f,z_end=%f, h_ini=%f,h_end=%f, bg_ini=%f,bg_end=%f, current emitter idx=%d\n", *(CRLBs_h + i * 5), *(fitting_para_h + i * 5), *(CRLBs_h + i * 5 + 1), *(fitting_para_h + i * 5 + 1), *(CRLBs_h + i * 5 + 2), *(fitting_para_h + i * 5 + 2), *(CRLBs_h + i * 5 + 3), *(fitting_para_h + i * 5 + 3), *(CRLBs_h + i * 5 + 4), *(fitting_para_h + i * 5 + 4), i + 1);
	}
	*/
	// .mat output
	double* fitting_para_crlb = new double[emitter_num * fit_para_num];
	double* fitting_para_end = new double[emitter_num * fit_para_num];
	double* fitting_para_ChiSq = new double[emitter_num];
	double* device_debug_out = new double[iterations*2 * emitter_num];
	memset(fitting_para_crlb, 0, emitter_num * fit_para_num * sizeof(double));
	memset(fitting_para_end, 0, emitter_num * fit_para_num * sizeof(double));
	memset(fitting_para_ChiSq, 0, emitter_num * sizeof(double));
	memset(device_debug_out, 0, iterations * 2 * emitter_num * sizeof(double));
	for (int i = 0; i < emitter_num; i++)
	{
		*(fitting_para_ChiSq + i) = (double)*(LogLikelihood_h + i);
		for (int j = 0; j < iterations*2; j++)
		{
			*(device_debug_out + i * iterations*2 + j) = (double)*(device_debug_h + i * iterations*2 + j);
		}
	}
	for (int i = 0; i < emitter_num * fit_para_num; i++)
	{
		*(fitting_para_crlb + i) = (double)*(CRLBs_h + i);
		*(fitting_para_end + i) = (double)*(fitting_para_h + i);
	}
	MATFile* pmat;
	mxArray* pa1;
	string file_crlb_full = seg_data_path + "crlb.mat";
	string file_fitting_para_full = seg_data_path + "fitting_result.mat";
	string finalChiSq_full = seg_data_path + "ChiSq.mat";
	string device_debug_out_char_full = seg_data_path + "device_debug_out.mat";
	const char* file_crlb = file_crlb_full.c_str();
	const char* file_fitting_para = file_fitting_para_full.c_str();
	const char* finalChiSq = finalChiSq_full.c_str();
	const char* device_debug_out_char = device_debug_out_char_full.c_str();

	pmat = matOpen(file_crlb, "w");
	pa1 = mxCreateDoubleMatrix(fit_para_num, emitter_num, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)fitting_para_crlb, fit_para_num * emitter_num * sizeof(double));
	matPutVariable(pmat, "crlb_results", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);

	pmat = matOpen(file_fitting_para, "w");
	pa1 = mxCreateDoubleMatrix(fit_para_num, emitter_num, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)fitting_para_end, fit_para_num * emitter_num * sizeof(double));
	matPutVariable(pmat, "fitting_results", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);

	pmat = matOpen(finalChiSq, "w");
	pa1 = mxCreateDoubleMatrix(emitter_num, 1, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)fitting_para_ChiSq, emitter_num * sizeof(double));
	matPutVariable(pmat, "ChiSq", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);

	pmat = matOpen(device_debug_out_char, "w");
	pa1 = mxCreateDoubleMatrix(iterations*2, emitter_num, mxREAL);
	memcpy((void*)(mxGetPr(pa1)), (void*)device_debug_out, emitter_num * iterations*2 * sizeof(double));
	matPutVariable(pmat, "test", pa1);
	mxDestroyArray(pa1);
	matClose(pmat);

	delete[] fitting_para_crlb, fitting_para_end, fitting_para_ChiSq, device_debug_out;
	// .mat output end
	delete[] coef_det_h, coef_exc_h, data_h, offset_map_h, var_map_h, gain_map_h, map_ptr_x_h, map_ptr_y_h, para_config_h;
	delete[] fitting_para_h, CRLBs_h, LogLikelihood_h, device_debug_h;
	/*
	cudasafe(cudaFree(coef_det_d), "cudaFree failed on coef_det_d.", __LINE__);
	cudasafe(cudaFree(coef_exc_d), "cudaFree failed on coef_exc_d.", __LINE__);
	cudasafe(cudaFree(data_d), "cudaFree failed on data_d.", __LINE__);
	cudasafe(cudaFree(offset_map_d), "cudaFree failed on offset_map_d.", __LINE__);
	cudasafe(cudaFree(var_map_d), "cudaFree failed on var_map_d.", __LINE__);
	cudasafe(cudaFree(gain_map_d), "cudaFree failed on gain_map_d.", __LINE__);
	cudasafe(cudaFree(map_ptr_x_d), "cudaFree failed on map_ptr_x_d.", __LINE__);
	cudasafe(cudaFree(map_ptr_y_d), "cudaFree failed on map_ptr_y_d.", __LINE__);
	cudasafe(cudaFree(fitting_para_d), "cudaFree failed on fitting_para_d.", __LINE__);
	cudasafe(cudaFree(CRLBs_d), "cudaFree failed on CRLBs_d.", __LINE__);
	cudasafe(cudaFree(LogLikelihood_d), "cudaFree failed on LogLikelihood_d.", __LINE__);
	cudasafe(cudaFree(device_debug_d), "cudaFree failed on device_debug_d.", __LINE__);
	cudasafe(cudaFree(para_config_d), "cudaFree failed on para_config_d.", __LINE__);
	*/
	cudasafe(cudaDeviceReset(), "sync failed", __LINE__);

}