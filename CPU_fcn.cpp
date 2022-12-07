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
	// Light sheet offset smoothing and interpolating in t
	vector<double> time_samp;
	vector<double> data_samp;
	vector<double> time_comp;
	vector<double> data_comp;
	//tk::spline smoothed_os;
	float cur_os;
	float sum;
	float counter;
	int ini_idx;
	int end_idx;
	int smooth_seg_size;
	//DoubleVec res;
	for (int i = 0; i < y_range; i++)
	{
		for (int j = 0; j < x_range; j++)
		{
			time_samp.clear();
			data_samp.clear();
			time_comp.clear();
			data_comp.clear();
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
				/*
				if (!isnan(mat_os_SM_smo[k][i][j]))
				{
					time_samp.push_back((double)k);
					data_samp.push_back((double)mat_os_SM_smo[k][i][j]);
				}
				time_comp.push_back((double)k);
				*/
			}

			/*
			if (data_samp.size()>10)
			{
				res = interpolation(time_samp, data_samp, time_comp);
				for (int k = 0; k < t_range; k++)
				{
					cur_os = res[k];
					mat_os_SM_smo[k][i][j] = (float)cur_os;
				}
			}
			else
			{
				for (int k = 0; k < t_range; k++)
					mat_os_SM_smo[k][i][j] = 0;
			}
			*/
		}
	}
	for (int i = 0; i < SM_num; i++)
	{
		int idx_t = *(map_ptr_t_h_SM + i) - 1;
		int idx_x = (int)(ceil(*(map_ptr_x_h_SM + i) / (xybinsize_SM / pixel_size_cam)) - 1);
		int idx_y = (int)(ceil(*(map_ptr_y_h_SM + i) / (xybinsize_SM / pixel_size_cam)) - 1);
		*(fitting_para_SM_h + i * fit_para_num + 5) = mat_os_SM_smo[idx_t][idx_y][idx_x];
	}
	return mat_os_SM_smo;
}
