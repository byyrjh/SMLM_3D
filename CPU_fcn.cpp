#include <vector>
#include "para_config.h"
#include "miscellaneous.h"
vector<vector<vector<float>>> LS_os_calc_SM(float* fitting_para_SM_h, float* CRLBs_SM_h, float* map_ptr_t_h_SM, float* map_ptr_x_h_SM, float* map_ptr_y_h_SM, float cam_map_size_ptr, float cam_map_size_y_ptr, float num_vol_ptr, float SM_num, float xybinsize_SM, int smooth_seg_SM)
{
	// smooth in t by hyperstack based averaging
	float x_range = ceil(cam_map_size_ptr / (xybinsize_SM / pixel_size_cam));
	float y_range = ceil(cam_map_size_y_ptr / (xybinsize_SM / pixel_size_cam));
	float t_range = num_vol_ptr;
	int num_vol = (int)num_vol_ptr;
	// kickout criterion 1 NAN 2 absolute LS offset (within 600) 3 crlb (this is a good metric within 25) 4 ChiSq (insignificant metric)
	// crlb = 25 is the real value. The conversion relationship is crlb_real = sqrt(crlb_raw)*10
	vector<vector<vector<float>>> mat_os_SM(num_vol, vector<vector<float>>(y_range, vector<float>(x_range)));
	vector<vector<vector<float>>> mat_counter_SM(num_vol, vector<vector<float>>(y_range, vector<float>(x_range)));
	vector<vector<vector<float>>> mat_os_SM_smo(num_vol, vector<vector<float>>(y_range, vector<float>(x_range)));
	// Light sheet offset register and averaging
	bool kickout;
	for (int i = 0; i < SM_num; i++) // offset registering
	{
		kickout = false;
		int idx_t = (*(map_ptr_t_h_SM + i)) - 1;
		int idx_x = (int)(ceil(*(map_ptr_x_h_SM + i) / (xybinsize_SM / pixel_size_cam)) - 1);
		int idx_y = (int)(ceil(*(map_ptr_y_h_SM + i) / (xybinsize_SM / pixel_size_cam)) - 1);
		kickout = kickout || isnan(*(fitting_para_SM_h + i * fit_para_num));
		kickout = kickout || isnan(*(fitting_para_SM_h + i * fit_para_num + 1));
		kickout = kickout || isnan(*(fitting_para_SM_h + i * fit_para_num + 2));
		kickout = kickout || isnan(*(fitting_para_SM_h + i * fit_para_num + 5));
		if (sqrt(*(CRLBs_SM_h + i * fit_para_num + 5))*10 > 25)
			kickout = true;
		if ((*(fitting_para_SM_h + i * fit_para_num + 5) < -60) || (*(fitting_para_SM_h + i * fit_para_num + 5) > 60)) //600/10 = 60
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
	for (int i = 0; i < t_range; i++) //light sheet offset averaging
	{
		for (int j = 0; j < y_range; j++)
		{
			for (int k = 0; k < x_range; k++)
			{
				if (mat_counter_SM[i][j][k] == 0)
					++mat_counter_SM[i][j][k]; // this step cannot avoid NAN, which originate from initialization of vector
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
					if ((mat_os_SM[ini_idx + m][i][j])!=0) // this step skip NANs
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
		*(fitting_para_SM_h + i * fit_para_num + 5) = mat_os_SM_smo[idx_t][idx_y][idx_x];
	}
	return mat_os_SM_smo; // returned vector contains sparse NANs
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

void LS_plane_fitting(float* LS_plane_coeff, float* LS_os_data, int x_range, int y_range, int t_range)
{
	vector<float> x_pos_temp;
	vector<float> y_pos_temp;
	vector<float> const_temp;
	vector<float> ls_os_temp;
	vector<vector<float>> A_mat;
	float* sq_mat = new float[9];
	float* sq_mat_inv = new float[9];
	int ptr_mat;
	int len_sum;
	float sum_element;
	for (int t = 0; t < t_range; t++)
	{
		x_pos_temp.clear();
		y_pos_temp.clear();
		ls_os_temp.clear();
		const_temp.clear();
		A_mat.clear();
		for (int y = 0; y < y_range; y++)
		{
			for (int x = 0; x < x_range; x++)
			{
				ptr_mat = t * y_range * x_range + y * y_range + x;
				if (!isnan(*(LS_os_data + ptr_mat)))
				{
					x_pos_temp.push_back(x);
					y_pos_temp.push_back(y);
					const_temp.push_back(1);
					ls_os_temp.push_back(*(LS_os_data + ptr_mat));
				}
			}
		}
		A_mat.push_back(x_pos_temp);
		A_mat.push_back(y_pos_temp);
		A_mat.push_back(const_temp);
		// generated matrix is symmetric
		len_sum = x_pos_temp.size();
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				sum_element = 0;
				for (int k = 0; k < len_sum;k++)
				{
					sum_element += A_mat[i][k] * A_mat[j][k];
				}
				sq_mat[j + i * 3] = sum_element;
			}
		}
		MatInvN(sq_mat, sq_mat_inv, 3);
		vector<vector<float>> rect_mat(3, vector<float>(len_sum));
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < len_sum; j++)
			{
				sum_element = 0;
				for (int k = 0; k < 3; k++)
				{
					sum_element += A_mat[k][j] * sq_mat_inv[i * 3 + k];
				}
				rect_mat[i][j] = sum_element;
			}
		}
		for (int i = 0; i < 3; i++)
		{
			sum_element = 0;
			for (int k = 0; k < len_sum; k++)
			{
				sum_element += rect_mat[i][k] * ls_os_temp[k];
			}
			//coefficient model ax + by + c = LS_os
			*(LS_plane_coeff + 3 * t + i) = sum_element;
		}
	}

}

void LS_plane_coeff_smooth(float* LS_plane_coeff_raw, float* LS_plane_coeff_smo, int smooth_range, int t_range)
{
	int ini_idx;
	int end_idx;
	int smooth_seg_size;
	float sum;
	for (int i = 0; i < 3; i++)
	{
		for (int k = 0; k < t_range; k++)
		{
			ini_idx = k - smooth_range / 2;
			end_idx = k + smooth_range / 2;
			if (ini_idx < 0)
			{
				smooth_seg_size = 1 + smooth_range / 2 + k;
				ini_idx = 0;
			}
			else if (end_idx > (t_range - 1))
				smooth_seg_size = smooth_range / 2 + t_range - k;
			else
				smooth_seg_size = 1 + smooth_range;
			sum = 0;
			for (int m = 0; m < smooth_seg_size; m++)
			{
				sum += LS_plane_coeff_raw[(ini_idx + m) * 3 + i];
				
			}
			LS_plane_coeff_smo[k * 3 + i] = sum / (float)smooth_seg_size;
		}
	}
}
void MatInvN(float* M, float* Minv, int sz)
{
	/*!
	 * \brief nxn partial matrix inversion
	 * \param M matrix to inverted
	 * \param Minv inverted matrix result
	 * \param DiagMinv just the inverted diagonal
	 * \param sz size of the matrix
	 */
	int ii, jj, kk, num, b;
	float tmp1 = 0;
	float* yy = new float[sz * sz];

	for (jj = 0; jj < sz; jj++) {
		//calculate upper matrix
		for (ii = 0; ii <= jj; ii++)
			//deal with ii-1 in the sum, set sum(kk=0->ii-1) when ii=0 to zero
			if (ii > 0) {
				for (kk = 0; kk <= ii - 1; kk++) tmp1 += M[ii + kk * sz] * M[kk + jj * sz];
				M[ii + jj * sz] -= tmp1;
				tmp1 = 0;
			}

		for (ii = jj + 1; ii < sz; ii++)
			if (jj > 0) {
				for (kk = 0; kk <= jj - 1; kk++) tmp1 += M[ii + kk * sz] * M[kk + jj * sz];
				M[ii + jj * sz] = (1 / M[jj + jj * sz]) * (M[ii + jj * sz] - tmp1);
				tmp1 = 0;
			}
			else { M[ii + jj * sz] = (1 / M[jj + jj * sz]) * M[ii + jj * sz]; }
	}

	tmp1 = 0;

	for (num = 0; num < sz; num++) {
		// calculate yy
		if (num == 0) yy[0] = 1;
		else yy[0] = 0;

		for (ii = 1; ii < sz; ii++) {
			if (ii == num) b = 1;
			else b = 0;
			for (jj = 0; jj <= ii - 1; jj++) tmp1 += M[ii + jj * sz] * yy[jj];
			yy[ii] = b - tmp1;
			tmp1 = 0;
		}

		// calculate Minv
		Minv[sz - 1 + num * sz] = yy[sz - 1] / M[(sz - 1) + (sz - 1) * sz];

		for (ii = sz - 2; ii >= 0; ii--) {
			for (jj = ii + 1; jj < sz; jj++) tmp1 += M[ii + jj * sz] * Minv[jj + num * sz];
			Minv[ii + num * sz] = (1 / M[ii + ii * sz]) * (yy[ii] - tmp1);
			tmp1 = 0;
		}
	}
	delete[] yy;
	

	return;

}

void LS_os_filter(float* LS_os_data_filter, float* LS_os_data, int x, int y, int t)
{
	double r_filter[9] = { sqrt(2), 1, sqrt(2), 1, 0, 1, sqrt(2), 1, sqrt(2) };
	double sigma_filter = 0.5;
	float filter_weight[9];
	double exp_in;
	float counter;
	float sum;
	int ptr_offset;
	int temp_idx;
	int temp_x;
	int temp_y;
	int conv_x;
	int conv_y;
	float temp_weight;
	for (int i = 0; i < 9; i++)
	{
		filter_weight[i] =(float)exp(-pow(r_filter[i], 2) / 2 / pow(sigma_filter, 2));
	}

	for (int data_idx = 0; data_idx < x * y * t; data_idx++)
	{
		counter = 0;
		sum = 0; 
		for (int k = 0; k < 3; k++)
		{
			ptr_offset = (k - 1) * x * y;
			if (data_idx + ptr_offset >= 0 && data_idx + ptr_offset < t * x * y)
			{
				temp_idx = (data_idx + ptr_offset) % (x * y);
				temp_y = temp_idx / y;
				temp_x = temp_idx % x;
				for (int j = 0; j < 3;j++)
				{
					for (int i = 0; i < 3; i++)
					{
						conv_y = temp_y - 1 + j;
						conv_x = temp_x - 1 + i;
						if (conv_y >= 0 && conv_y < y && conv_x >= 0 && conv_x < x)
						{
							temp_idx = data_idx + ptr_offset + (j - 1) * y + (i - 1);
							temp_weight = filter_weight[j * 3 + i];
							if (!isnan(*(LS_os_data+temp_idx))) // this condition still cannot avoid NANs
							{
								sum += *(LS_os_data + temp_idx) * temp_weight;
								counter += temp_weight;
							}
						}
					}
				}
			}
		}
		*(LS_os_data_filter + data_idx) = sum / counter;
	}
}
