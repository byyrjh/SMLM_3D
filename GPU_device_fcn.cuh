//**************************************************************************************************************************************************
// This function for calculation of the common term for Cspline is adpopted from
//"Analyzing Single Molecule Localization Microscopy Data Using Cubic Splines", Hazen Babcok, Xiaowei Zhuang,Scientific Report, 1, 552 , 2017.

#include "GPU_Lib.h"
#include "para_config.h"
#include <stdio.h>
#include <cstdlib>
#include <cstring>

__device__ inline void kernel_bg_eval(const float* data_cur, const float* offset_map, const float* gain_map, const float* pos_x, const float* pos_y, float* NewTheta)
{
	float max_num_pho = 0;
	float min_num_pho = 0;
	for (int kk = 0; kk < slice_num; kk++)
	{
		for (int ii = 0; ii < seg_size; ii++) for (int jj = 0; jj < seg_size; jj++) // jj is inner loop
		{
			int i = jj + ii * seg_size;
			int rem_idx = (i + 1) % seg_size;
			if (rem_idx == 0) rem_idx = seg_size;
			int cur_idx = (static_cast<int>(*pos_y) - ((seg_size - 1) / 2 + 1) + (i + 1 - rem_idx) / seg_size) * cam_map_size + static_cast<int>(*pos_x) - ((seg_size - 1) / 2 + 1) + rem_idx - 1;
			float cur_offset = *(offset_map + cur_idx);   //calculate map index
			float cur_gain = *(gain_map + cur_idx);
			max_num_pho = fmaxf(max_num_pho, (data_cur[kk * seg_size * seg_size + seg_size * ii + jj] - cur_offset) / cur_gain);
			min_num_pho = fminf(min_num_pho, (data_cur[kk * seg_size * seg_size + seg_size * ii + jj] - cur_offset) / cur_gain);
			if (min_num_pho < 0) min_num_pho = 0;
		}
	}
	*(NewTheta + 3) = max_num_pho;
	*(NewTheta + 4) = min_num_pho;
}

__device__ inline void kernel_h_bg_init(const float* data_cur, const float* offset_map, const float* gain_map, const float* pos_x, const float* pos_y, float* NewTheta)
{
	//        calculate histogram  and initialize h and bg
	float max_num_pho = *(NewTheta + 3);
	float min_num_pho = *(NewTheta + 4);
	int* hist_dens = new int[binsize];
	for (int i = 0; i < binsize; i++)
		*(hist_dens + i) = 0;
	float bin_width = (max_num_pho - min_num_pho) / binsize;
	//calculate map index
	for (int kk = 0; kk < slice_num; kk++) for (int ii = 0; ii < seg_size; ii++)for (int jj = 0; jj < seg_size; jj++)
	{
		int rem_idx = (jj + ii * seg_size + 1) % seg_size;
		if (rem_idx == 0) rem_idx = seg_size;
		int cur_idx = (static_cast<int>(*pos_y) - ((seg_size - 1) / 2 + 1) + (jj + ii * seg_size + 1 - rem_idx) / seg_size) * cam_map_size + static_cast<int>(*pos_x) - ((seg_size - 1) / 2 + 1) + rem_idx - 1;
		float cur_offset = *(offset_map + cur_idx);
		float cur_gain = *(gain_map + cur_idx);
		int bin_pos = 0;
		float num_pho = (*(data_cur + kk * seg_size * seg_size + seg_size * ii + jj) - cur_offset) / cur_gain;
		if (num_pho > 0)	bin_pos = (int)lrintf((num_pho - min_num_pho) / bin_width);
		++* (hist_dens + bin_pos);
	}
	float hist_cum = 0;
	int h_idx = 0;
	int bg_idx = 0;
	int occurre = 0;
	for (int i = 0; i < binsize; i++)
	{
		hist_cum += ((float)*(hist_dens + binsize - 1 - i)) / (seg_size * seg_size * slice_num);
		if (hist_cum > p_value && h_idx == 0) h_idx = i;
		if (*(hist_dens + binsize - 1 - i) > occurre)
		{
			occurre = *(hist_dens + binsize - 1 - i);
			bg_idx = i;
		}
	}
	*(NewTheta + 4) = max_num_pho - bin_width * static_cast<float>(bg_idx + 0.5);
	*(NewTheta + 3) = max_num_pho - bin_width * static_cast<float>(h_idx + 0.5) - *(NewTheta + 4);
	delete[] hist_dens;
}

__device__ inline void kernel_xy_init(const float* data_cur, const float* offset_map, const float* gain_map, const float* pos_x, const float* pos_y, float* NewTheta)
{

	float* tmpx = new float[1];
	float* tmpy = new float[1];
	float* tmpz = new float[1];
	float* tmpsum = new float[1];
	*tmpx = 0;
	*tmpy = 0;
	*tmpz = 0;
	*tmpsum = 0;
	const float bg = *(NewTheta + 4);


	for (int kk = 0; kk < slice_num; kk++)
	{
		for (int ii = 0; ii < seg_size; ii++) for (int jj = 0; jj < seg_size; jj++) // jj is inner loop
		{
			int i = jj + ii * seg_size;
			int rem_idx = (i + 1) % seg_size;
			if (rem_idx == 0) rem_idx = seg_size;
			int cur_idx = (static_cast<int>(*pos_y) - ((seg_size - 1) / 2 + 1) + (i + 1 - rem_idx) / seg_size) * cam_map_size + static_cast<int>(*pos_x) - ((seg_size - 1) / 2 + 1) + rem_idx - 1;
			float cur_offset = *(offset_map + cur_idx);   //calculate map index
			float cur_gain = *(gain_map + cur_idx);
			*tmpy += fmaxf((data_cur[kk * seg_size * seg_size + seg_size * ii + jj] - cur_offset) / cur_gain - bg, 0) * (ii + 1);
			*tmpx += fmaxf((data_cur[kk * seg_size * seg_size + seg_size * ii + jj] - cur_offset) / cur_gain - bg, 0) * (jj + 1);
			*tmpsum += fmaxf((data_cur[kk * seg_size * seg_size + seg_size * ii + jj] - cur_offset) / cur_gain - bg, 0);
		}
	}

	if (abs(*tmpx / *tmpsum - ((seg_size - 1) / 2 + 1)) <= init_esti_xy)
		*NewTheta = *tmpx / *tmpsum - ((seg_size - 1) / 2 + 1);
	else
		*NewTheta = 0;
	if (abs(*tmpy / *tmpsum - ((seg_size - 1) / 2 + 1)) <= init_esti_xy)
		*(NewTheta + 1) = *tmpy / *tmpsum - ((seg_size - 1) / 2 + 1);
	else
		*(NewTheta + 1) = 0;

	delete[] tmpx;
	delete[] tmpy;
	delete[] tmpz;
	delete[] tmpsum;
}

__device__ inline void kernel_z_init(const float* data_cur, const float* offset_map, const float* gain_map, const float* pos_x, const float* pos_y, float* NewTheta, const float* lat_inten_cali_d)
{
	float x_pos = *NewTheta;//jj
	float y_pos = *(NewTheta + 1);//ii
	float z_pos = 0;
	for (int kk = 0; kk < (slice_num - 1) / 2; kk++)
	{
		float upper_mo = 0;
		float lower_mo = 0;
		for (int ii = 0; ii < seg_size; ii++) for (int jj = 0; jj < seg_size; jj++) // jj is inner loop
		{
			const float bg = *(NewTheta + 4);
			int i = jj + ii * seg_size;
			int rem_idx = (i + 1) % seg_size;
			if (rem_idx == 0) rem_idx = seg_size;
			int cur_idx = (static_cast<int>(*pos_y) - ((seg_size - 1) / 2 + 1) + (i + 1 - rem_idx) / seg_size) * cam_map_size + static_cast<int>(*pos_x) - ((seg_size - 1) / 2 + 1) + rem_idx - 1;
			float cur_offset = *(offset_map + cur_idx);   //calculate map index
			float cur_gain = *(gain_map + cur_idx);
			upper_mo += fmaxf((data_cur[kk * seg_size * seg_size + seg_size * ii + jj] - cur_offset) / cur_gain - bg, 0);
			lower_mo += fmaxf((data_cur[(slice_num - 1 - kk) * seg_size * seg_size + seg_size * ii + jj] - cur_offset) / cur_gain - bg, 0);
		}
		float err_sq = 10;
		int z_id = 0;
		for (int i = 0; i < LS_stepsize / step_size; i++)
		{
			if ((*(lat_inten_cali_d + i + kk * LS_stepsize / step_size) - upper_mo / lower_mo) * (*(lat_inten_cali_d + i + kk * LS_stepsize / step_size) - upper_mo / lower_mo) < err_sq)
			{
				z_id = i;
				err_sq = (*(lat_inten_cali_d + i + kk * LS_stepsize / step_size) - upper_mo / lower_mo) * (*(lat_inten_cali_d + i + kk * LS_stepsize / step_size) - upper_mo / lower_mo);
			}
		}
		z_pos += (LS_stepsize / step_size / 2 - z_id) / ((slice_num - 1) / 2);
	}
	*(NewTheta + 2) = z_pos;
}

__device__ inline void kernel_computeDelta3D_det(float x_delta, float y_delta, float z_delta, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf)
{

	int i, j, k;
	float cx, cy, cz;

	cz = 1.0;
	for (i = 0; i < 4; i++) {
		cy = 1.0;
		for (j = 0; j < 4; j++) {
			cx = 1.0;
			for (k = 0; k < 4; k++) {
				delta_f[i * 16 + j * 4 + k] = cz * cy * cx;
				if (k < 3) {
					delta_dxf[i * 16 + j * 4 + k + 1] = ((float)k + 1) * cz * cy * cx;
				}

				if (j < 3) {
					delta_dyf[i * 16 + (j + 1) * 4 + k] = ((float)j + 1) * cz * cy * cx;
				}

				if (i < 3) {
					delta_dzf[(i + 1) * 16 + j * 4 + k] = ((float)i + 1) * cz * cy * cx;
				}

				cx = cx * x_delta;
			}
			cy = cy * y_delta;
		}
		cz = cz * z_delta;
	}
}

__device__ inline void kernel_computeDelta3D_exc(float z_exc_frac, float* delta_g, float* delta_dzg)
{
	float cz = 1;
	for (int i = 0; i < 4; i++)
	{
		delta_g[i] = cz;
		if (i < 3)
		{
			delta_dzg[i + 1] = ((float)i + 1) * cz;
		}
		cz = cz * z_exc_frac;
	}
}

__device__ inline void kernel_DerivativeSpline(bool offset_fit, int* x_spl, int* y_spl, int* z_spl, int* z_spl_exc, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf, float* delta_g, float* delta_dzg, const float* coef_det_d, const float* coef_exc_d, float* theta, float* dudt)
{
	float temp = 0, temp_g = 0, temp_dzg = 0;
	if (offset_fit)
		memset(dudt, 0, fit_para_num * sizeof(float));
	else
		memset(dudt, 0, (fit_para_num - 1) * sizeof(float));
	for (int i = 0; i < 64; i++) {
		temp += delta_f[i] * coef_det_d[i * (spline_x * spline_y * spline_z) + (*z_spl) * (spline_x * spline_y) + (*y_spl) * spline_y + *x_spl];
		dudt[0] += delta_dxf[i] * coef_det_d[i * (spline_x * spline_y * spline_z) + (*z_spl) * (spline_x * spline_y) + (*y_spl) * spline_y + *x_spl];
		dudt[1] += delta_dyf[i] * coef_det_d[i * (spline_x * spline_y * spline_z) + (*z_spl) * (spline_x * spline_y) + (*y_spl) * spline_y + *x_spl];
		dudt[2] += delta_dzf[i] * coef_det_d[i * (spline_x * spline_y * spline_z) + (*z_spl) * (spline_x * spline_y) + (*y_spl) * spline_y + *x_spl];
	}
	for (int i = 0; i < 4; i++)
	{
		temp_g += delta_g[i] * coef_exc_d[(*z_spl_exc) * num_coef_per_pix_axial + i];
		temp_dzg += delta_dzg[i] * coef_exc_d[(*z_spl_exc) * num_coef_per_pix_axial + i];
	}
	dudt[0] *= -1.0f * theta[3] * temp_g;  //dmu/dx  dmu/dy  dmu/dz  dmu/dh  dmu/dbg dmu/dz_offset
	dudt[1] *= -1.0f * theta[3] * temp_g;  //Theta:= x y z h bg z_offset
	dudt[2] = -1.0f * theta[3] * temp_g * dudt[2] - theta[3] * temp * temp_dzg;
	dudt[3] = temp * temp_g;
	dudt[4] = 1.0f;
	if (offset_fit) dudt[5] = -1.0f * theta[3] * temp * temp_dzg;
}

__device__ inline int kernel_cholesky(float* A, int n, float* L, float* U) //The Cholesky–Banachiewicz and Cholesky–Crout algorithms
{
	int info = 0;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < (i + 1); j++) {
			float s = 0;
			for (int k = 0; k < j; k++)
				s += U[i * n + k] * U[j * n + k];
			if (i == j) {
				if (A[i * n + i] - s >= 0) {
					U[i * n + j] = sqrt(A[i * n + i] - s);
					L[j * n + i] = U[i * n + j];
				}
				else {
					info = 1;
					return info;
				}
			}
			else {
				U[i * n + j] = (1.0 / U[j * n + j] * (A[i * n + j] - s));
				L[j * n + i] = U[i * n + j];
			}
		}
	return info;
}

__device__ inline void kernel_luEvaluate(float* L, float* U, float* b, const int n, float* x)
{
	//Ax = b -> LUx = b. Then y is defined to be Ux
	//for sigmaxy, we have 6 parameters
	float* y = new float[n];
	memset(y, 0, n * sizeof(float));
	// Forward solve Ly = b
	for (int i = 0; i < n; i++)
	{
		y[i] = b[i];
		for (int j = 0; j < i; j++)
		{
			y[i] -= L[j * n + i] * y[j];
		}
		y[i] /= L[i * n + i];
	}
	// Backward solve Ux = y
	for (int i = n - 1; i >= 0; i--)
	{
		x[i] = y[i];
		for (int j = i + 1; j < n; j++)
		{
			x[i] -= U[j * n + i] * x[j];
		}
		x[i] /= U[i * n + i];
	}
	delete[] y;
}

__device__ inline void kernel_MatInvN(float* M, float* Minv, float* DiagMinv, int sz)
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
	if (DiagMinv) for (ii = 0; ii < sz; ii++) DiagMinv[ii] = Minv[ii * sz + ii];

	return;

}

