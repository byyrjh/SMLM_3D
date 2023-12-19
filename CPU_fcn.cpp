#include <vector>
#include "para_config.h"
#include "miscellaneous.h"
vector<vector<vector<float>>> LS_os_calc_SM(float* fitting_para_SM_h, float* map_ptr_t_h_SM, float* map_ptr_x_h_SM, float* map_ptr_y_h_SM, float cam_map_size_ptr, float cam_map_size_y_ptr, float num_vol_ptr, float SM_num, float xybinsize_SM, int smooth_seg_SM)
{
	float x_range = ceil(cam_map_size_ptr / (xybinsize_SM / pixel_size_cam));
	float y_range = ceil(cam_map_size_y_ptr / (xybinsize_SM / pixel_size_cam));
	float t_range = num_vol_ptr;
	int num_vol = (int)num_vol_ptr;

	vector<vector<vector<float>>> mat_os_SM(t_range, vector<vector<float>>(y_range, vector<float>(x_range)));
	vector<vector<vector<float>>> mat_counter_SM(t_range, vector<vector<float>>(y_range, vector<float>(x_range)));
	vector<vector<vector<float>>> mat_os_SM_smo(t_range, vector<vector<float>>(y_range, vector<float>(x_range)));
	// Light sheet offset register and averaging
	bool kickout;
	for (int i = 0; i < SM_num; i++) // offset registering
	{
		kickout = false;
		int idx_t = *(map_ptr_t_h_SM + i) - 1;
		int idx_x = (int)(ceil(*(map_ptr_x_h_SM + i) / (xybinsize_SM / pixel_size_cam)) - 1);
		int idx_y = (int)(ceil(*(map_ptr_y_h_SM + i) / (xybinsize_SM / pixel_size_cam)) - 1);
		kickout = kickout || isnan(*(fitting_para_SM_h + i * fit_para_num));
		kickout = kickout || isnan(*(fitting_para_SM_h + i * fit_para_num + 1));
		kickout = kickout || isnan(*(fitting_para_SM_h + i * fit_para_num + 2));
		kickout = kickout || isnan(*(fitting_para_SM_h + i * fit_para_num + 5));
		if ((*(fitting_para_SM_h + i * fit_para_num + 5) < -100) || (*(fitting_para_SM_h + i * fit_para_num + 5) > 100))
			kickout = true;
		if (!kickout)
		{
			float x_pos = *(map_ptr_x_h_SM + i) + *(fitting_para_SM_h + i * fit_para_num);
			float y_pos = *(map_ptr_y_h_SM + i) + *(fitting_para_SM_h + i * fit_para_num + 1);
			kickout = kickout || x_pos<0 || x_pos>cam_map_size_ptr;
			kickout = kickout || y_pos<0 || y_pos>cam_map_size_y_ptr;
			if (!kickout)
			{
				mat_os_SM[idx_t][idx_y][idx_x] += *(fitting_para_SM_h + i * fit_para_num + 5);
				mat_counter_SM[idx_t][idx_y][idx_x] += 1;
			}
		}
	}
	for (int i = 0; i < t_range; i++) // offset averaging
	{
		for (int j = 0; j < y_range; j++)
		{
			for (int k = 0; k < x_range; k++)
			{
				if (mat_counter_SM[i][j][k] == 0)
					++mat_counter_SM[i][j][k];
				mat_os_SM[i][j][k] = mat_os_SM[i][j][k] / mat_counter_SM[i][j][k];
			}
		}
	}
	// Light sheet offset smoothing in t
	float cur_os;
	float sum;
	float counter;
	int ini_idx;
	int end_idx;
	int smooth_seg_size;

	for (int i = 0; i < y_range; i++)
	{
		for (int j = 0; j < x_range; j++)
		{
			for (int k = 0; k < t_range; k++)
			{
				ini_idx = k - smooth_seg_SM / 2;
				end_idx = k + smooth_seg_SM / 2;
				if (ini_idx < 0)
				{
					smooth_seg_size = 1 + smooth_seg_SM/2 + k;
					ini_idx = 0;
				}
				else if (end_idx > (num_vol - 1))
					smooth_seg_size = smooth_seg_SM/2 + num_vol - k;
				else
					smooth_seg_size = 1 + smooth_seg_SM;
				sum = 0;
				counter = 0;
				for (int m = 0; m < smooth_seg_size; m++)
				{
					if ((mat_os_SM[ini_idx + m][i][j])!=0)
					{
						sum += mat_os_SM[ini_idx + m][i][j];
						++counter;
					}
				}
				mat_os_SM_smo[k][i][j] = sum / counter;
			}
		}
	}
	for (int i = 0; i < SM_num; i++)
	{
		int idx_t = *(map_ptr_t_h_SM + i) - 1;
		int idx_x = (int)(ceil(*(map_ptr_x_h_SM + i) / (xybinsize_SM / pixel_size_cam)) - 1);
		int idx_y = (int)(ceil(*(map_ptr_y_h_SM + i) / (xybinsize_SM / pixel_size_cam)) - 1);
		//*(fitting_para_SM_h + i * fit_para_num + 5) = mat_os_SM_smo[idx_t][idx_y][idx_x];
	}
	return mat_os_SM_smo;
}

void LS_os_calc_FM(float* fitting_para_h, double* test_LS_os, int& num_vol, float& stationary_pos_ptr, float& vol_per_hyper_ptr, int& smooth_seg, int& FM_trace)
{
	int vol_per_hyper = vol_per_hyper_ptr - stationary_pos_ptr + 1;
	int num_hyperstack = num_vol / vol_per_hyper;
	int smooth_seg_size;
	int ini_idx;
	int end_idx;
	int cur_FM_idx;
	for (int cur_hyper = 0; cur_hyper < num_hyperstack; cur_hyper++)
	{
		for (int j = 0; j < vol_per_hyper; j++)
		{
			ini_idx = j - smooth_seg / 2;
			ini_idx = std::max(ini_idx, 0);
			end_idx = std::min(j + smooth_seg / 2, vol_per_hyper - 1);
			smooth_seg_size = end_idx - ini_idx + 1;
			for (int FM_idx = 0; FM_idx < FM_trace; FM_idx++)// FM data structure 11111 22222 33333 44444 55555 .... 12345 represent time(volume indices) 5 FMs are sorted in the same way for each block
			{
				float LS_os = 0;
				float os_counter = 0;
				int cur_idx;
				cur_FM_idx = (cur_hyper * vol_per_hyper + j) * FM_trace + FM_idx;
				for (int cur_smooth_seg = 0; cur_smooth_seg < smooth_seg_size; cur_smooth_seg++)
				{
					cur_idx = (cur_hyper * vol_per_hyper + ini_idx + cur_smooth_seg) * FM_trace + FM_idx;
					//printf("current index is %d, current value is %f", cur_idx, *(fitting_para_h + cur_idx * fit_para_num + 5));
					if (!isnan(*(fitting_para_h + cur_idx * fit_para_num + 5)))
					{
						LS_os += *(fitting_para_h + cur_idx * fit_para_num + 5);
						++os_counter;
					}
				}
				
				*(fitting_para_h + cur_FM_idx * fit_para_num + 5) = LS_os / os_counter;
				*(test_LS_os + cur_FM_idx) = (double)*(fitting_para_h + cur_FM_idx * fit_para_num + 5);
			}
		}
	}
}
