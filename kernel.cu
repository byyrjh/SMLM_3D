#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 
#include <stdio.h>
#include "GPU_device_fcn.cuh"
#include "para_config.h"
#include <time.h>
#include <math.h>
#include <cmath>
#include "shared_struc.h"

__global__ void kernel_cuda_fitting(fitting_config* para_config, const float* coef_det_d, const float* coef_exc_d, const float* data_d, const float* offset_map_d, const float* var_map_d,
	const float* gain_map_d, const float* map_ptr_x_d, const float* map_ptr_y_d, float* fitting_para_d, float* CRLBs_d, float* LogLikelihood_d, float* device_debug_d)
{
	const int idx = (blockIdx.x) * block_size + threadIdx.x;
	//printf("hello from device, number of emitter=%d\n", *(para_config + 1));
	if (idx >= para_config->num_launch_thre) return;
	//if (idx == 11)
	if (idx < para_config->num_launch_thre)
	{
		float offset_global = 0;
		bool offset_fit;
		if (para_config->num_fitting_para == 6)             // config   num_fitting_para(5 or 6)   initial emitter idx    number of emitters to fit
			offset_fit = true;
		else
			offset_fit = false;
		float NewTheta[fit_para_num];//            x y z h bg z_offset
		float OldTheta[fit_para_num];//            x y z h bg z_offset
		const float pos_x = *(map_ptr_x_d + idx);
		const float pos_y = *(map_ptr_y_d + idx);
		const float* data_cur = data_d + idx * seg_size * seg_size * para_config->num_slice;
		float xc_int, yc_int, zc_int, xc_frac, yc_frac, zc_frac;
		float z_exc_int, z_exc_frac;
		float* Maxjump = new float[para_config->num_fitting_para];// = { 1,1,12,9,1.5,10 };  x y z max value in simu/3*2   h bg max value in simu/2/3*2 
		*Maxjump = 1;
		*(Maxjump + 1) = 1;
		*(Maxjump + 2) = 12;
		*(Maxjump + 3) = 9;
		*(Maxjump + 4) = 1.5;
		if (offset_fit) *(Maxjump + 5) = 10;
		const int offset_data_map = (spline_x + 1 - seg_size) / 2;
		float* jacobian = new float[para_config->num_fitting_para];
		memset(jacobian, 0, para_config->num_fitting_para * sizeof(float));
		float* hessian = new float[para_config->num_fitting_para * para_config->num_fitting_para];
		memset(hessian, 0, para_config->num_fitting_para * para_config->num_fitting_para * sizeof(float));
		
		float delta_f[64] = { 0 }, delta_dxf[64] = { 0 }, delta_dyf[64] = { 0 }, delta_dzf[64] = { 0 };  // mu=h*f*g+bg, f=:psf_det, g=:psf_exc
		float delta_g[4] = { 0 }, delta_dzg[4] = { 0 };
		float* NewDudt = new float[para_config->num_fitting_para]; //dmu/dx  dmu/dy  dmu/dz  dmu/dh  dmu/dbg dmu/dz_offset
		memset(NewDudt, 0, para_config->num_fitting_para * sizeof(float));
		float model, data;
		float NewChiSq;
		float OldChiSq = 1e11;
		
		float ChiSq_min = 0;
		int errFlag = 0;
		int iter;
		
		float* NewUpdate = new float[para_config->num_fitting_para];  //delta x y z h bg z_offset
		float* OldUpdate = new float[para_config->num_fitting_para];

		for (int i = 0; i < para_config->num_fitting_para; i++)
		{
			*(NewUpdate + i) = 1e11;
			*(OldUpdate + i) = 1e11;
		}
		
		float NewLambda = INIT_LAMBDA, OldLambda = INIT_LAMBDA, mu;
		
		float* L = new float[para_config->num_fitting_para * para_config->num_fitting_para];
		float* U = new float[para_config->num_fitting_para * para_config->num_fitting_para];
		float* M = new float[para_config->num_fitting_para * para_config->num_fitting_para];
		float* Minv = new float[para_config->num_fitting_para * para_config->num_fitting_para];
		float* Diag = new float[para_config->num_fitting_para];
		memset(L, 0, para_config->num_fitting_para * para_config->num_fitting_para * sizeof(float));
		memset(U, 0, para_config->num_fitting_para * para_config->num_fitting_para * sizeof(float));
		//delete[] L, U, M, Minv, Diag;
		

		if (offset_fit)
		{
			kernel_bg_eval(data_cur, offset_map_d, gain_map_d, &pos_x, &pos_y, NewTheta, para_config->num_slice, para_config->cam_x);
			kernel_h_bg_init(data_cur, offset_map_d, gain_map_d, &pos_x, &pos_y, NewTheta, para_config->num_slice, para_config->cam_x);
			kernel_xy_init(data_cur, offset_map_d, gain_map_d, &pos_x, &pos_y, NewTheta, para_config->num_slice, para_config->cam_x);
			*(NewTheta + 2) = 0.001;
			*(NewTheta + 5) = 0.001;
		}
		else
		{
			for (int i = 0; i < 6; i++)
				if (!isnan(*(fitting_para_d + idx * fit_para_num + i)))
					NewTheta[i] = *(fitting_para_d + idx * fit_para_num + i);
				else
					NewTheta[i] = 0.001;
			// fitting SM using offset of FM light sheet 
			/*
			if (para_config->num_slice == 5)
			{
				for (int i = 0; i < 6; i++)
					if (!isnan(*(fitting_para_d + idx * fit_para_num + i)))
						NewTheta[i] = *(fitting_para_d + idx * fit_para_num + i);
					else
						NewTheta[i] = 0.001;
			}
			else
			{
				*(NewTheta + 2) = 0.001;
				*(NewTheta + 5) = *(fitting_para_d + idx * fit_para_num + 5);
			}
			*/
		}
		
		//kernel_z_init(data_cur, offset_map_d, gain_map_d, &pos_x, &pos_y, NewTheta, lat_inten_cali_d);
		
		for (int ii = 0; ii < para_config->num_fitting_para; ii++) OldTheta[ii] = NewTheta[ii];
		xc_int = floor(-NewTheta[0]);
		xc_frac = -NewTheta[0] - xc_int;            // units of xc_int yc_int zc_int z_exc_int are PSF template pixel(lateral) and step(axial) 
		yc_int = floor(-NewTheta[1]);
		yc_frac = -NewTheta[1] - yc_int;
		zc_int = floor(-NewTheta[2]);				// unit of z_offset and z is number of steps in step_size
		zc_frac = -NewTheta[2] - zc_int;
		kernel_computeDelta3D_det(xc_frac, yc_frac, zc_frac, delta_f, delta_dxf, delta_dyf, delta_dzf);
		z_exc_int = floor(-NewTheta[2] - NewTheta[5]);
		z_exc_frac = -NewTheta[2] - NewTheta[5] - z_exc_int;
		kernel_computeDelta3D_exc(z_exc_frac, delta_g, delta_dzg);

		
		
		for (int kk = 0; kk < para_config->num_slice; kk++)for (int ii = 0; ii < seg_size; ii++) for (int jj = 0; jj < seg_size; jj++)
		{  // calculate for each pixel. initialize alpha beta and ChiSq
			int i = jj + ii * seg_size;
			int rem_idx = (i + 1) % seg_size;
			if (rem_idx == 0) rem_idx = seg_size;
			int cur_idx = (static_cast<int>(pos_y) - ((seg_size - 1) / 2 + 1) + (i + 1 - rem_idx) / seg_size) * para_config->cam_x + static_cast<int>(pos_x) - ((seg_size - 1) / 2 + 1) + rem_idx - 1;
			float cur_offset = *(offset_map_d + cur_idx);   //calculate map index
			float cur_gain = *(gain_map_d + cur_idx);
			float cur_var = *(var_map_d + cur_idx);
			int x_spl = jj + offset_data_map + xc_int;
			int y_spl = ii + offset_data_map + yc_int;
			int z_spl = spline_z / 2 + 1 + zc_int + LS_stepsize / step_size * (kk - (int)floorf(para_config->num_slice / 2)) - 1;// map z-zc to PSFdet template
			int z_spl_exc = spline_z / 2 + 1 + z_exc_int + LS_stepsize / step_size * (kk - (int)floorf(para_config->num_slice / 2)) - 1;  //map z-zc-z_offset to PSFexc template
			if ((x_spl < 0) || (x_spl > spline_x - 1)) continue;
			if ((y_spl < 0) || (y_spl > spline_y - 1)) continue;
			if ((z_spl < 0) || (z_spl > spline_z - 1)) continue;
			if ((z_spl_exc < 0) || (z_spl_exc > spline_z - 1)) continue;
			kernel_DerivativeSpline(offset_fit, &x_spl, &y_spl, &z_spl, &z_spl_exc, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_g, delta_dzg, coef_det_d, coef_exc_d, NewTheta, NewDudt);
			model = NewDudt[3] * NewTheta[3] + NewTheta[4] + cur_var / cur_gain / cur_gain;
			data = (*(data_cur + kk * seg_size * seg_size + ii * seg_size + jj) - cur_offset) / cur_gain + cur_var / cur_gain / cur_gain;
			if ((data > 0) && (model > 0))
			{
				NewChiSq += 2 * ((model - data) - data * log(model / data));
			}
			else
			{
				continue;
			}
			for (int k = 0; k < para_config->num_fitting_para; k++) jacobian[k] -= (1 - data / model) * NewDudt[k];

			for (int j = 0; j < para_config->num_fitting_para; j++) for (int k = j; k < para_config->num_fitting_para; k++)
			{
				hessian[j * para_config->num_fitting_para + k] += data / model / model * NewDudt[j] * NewDudt[k];
				hessian[k * para_config->num_fitting_para + j] = hessian[j * para_config->num_fitting_para + k];
			}
		}  //end to alpha beta and ChiSq initialization
		
		
		
		//********************** main iterative loop ****************************
		int pix_count = 0;
		for (iter = 0; iter < iterations; iter++)
		{
			if (abs((NewChiSq - OldChiSq) / NewChiSq) < TOLERANCE)
			{ // ChiSq converged
				break;
			}
			else
			{
				if (NewChiSq > ACCEPTANCE * OldChiSq)  // lambda selection has to be investigated
				{// ChiSq diverges. roll back to old parameters. re-try larger lambda augmented by a factor of 10
					for (int i = 0; i < para_config->num_fitting_para; i++)
					{
						NewTheta[i] = OldTheta[i];
						NewUpdate[i] = OldUpdate[i];
					}
					NewLambda = OldLambda;
					NewChiSq = OldChiSq;
					mu = fmaxf((1 + NewLambda * SCALE_UP) / (1 + NewLambda), 1.3f); // 
					NewLambda = SCALE_UP * NewLambda;
				}
				else if (NewChiSq < OldChiSq && errFlag == 0)
				{ // ChiSq is converging
					NewLambda = SCALE_DOWN * NewLambda;
					mu = 1 + NewLambda;
				}

				for (int i = 0; i < para_config->num_fitting_para; i++)
				{
					hessian[i * para_config->num_fitting_para + i] = hessian[i * para_config->num_fitting_para + i] * mu;
				}
				memset(L, 0, para_config->num_fitting_para * para_config->num_fitting_para * sizeof(float));
				memset(U, 0, para_config->num_fitting_para * para_config->num_fitting_para * sizeof(float));
				errFlag = kernel_cholesky(hessian, para_config->num_fitting_para, L, U);
				if (errFlag == 0)
				{
					for (int i = 0; i < para_config->num_fitting_para; i++)
					{
						OldTheta[i] = NewTheta[i];
						OldUpdate[i] = NewUpdate[i];
					}
					OldLambda = NewLambda;
					OldChiSq = NewChiSq;
					kernel_luEvaluate(L, U, jacobian, para_config->num_fitting_para, NewUpdate);
					//updateFitParameters
					for (int ll = 0; ll < para_config->num_fitting_para; ll++)
					{
						if (NewUpdate[ll] / OldUpdate[ll] < -0.7f) Maxjump[ll] = Maxjump[ll] * 0.7;
						NewUpdate[ll] = NewUpdate[ll] / (1 + fabs(NewUpdate[ll] / Maxjump[ll]));  //  harmonic mean/2
						NewTheta[ll] = NewTheta[ll] + NewUpdate[ll];
					}
					//updateFitValues
					xc_int = floor(-NewTheta[0]);
					xc_frac = -NewTheta[0] - xc_int;
					yc_int = floor(-NewTheta[1]);
					yc_frac = -NewTheta[1] - yc_int;
					zc_int = floor(-NewTheta[2]);
					zc_frac = -NewTheta[2] - zc_int;
					z_exc_int = floor(-NewTheta[2] - NewTheta[5]);
					z_exc_frac = -NewTheta[2] - NewTheta[5] - z_exc_int;
					NewChiSq = 0;
					memset(jacobian, 0, para_config->num_fitting_para * sizeof(float));
					memset(hessian, 0, para_config->num_fitting_para * para_config->num_fitting_para * sizeof(float));
					kernel_computeDelta3D_exc(z_exc_frac, delta_g, delta_dzg);
					kernel_computeDelta3D_det(xc_frac, yc_frac, zc_frac, delta_f, delta_dxf, delta_dyf, delta_dzf);
					for (int kk = 0; kk < para_config->num_slice; kk++)for (int ii = 0; ii < seg_size; ii++) for (int jj = 0; jj < seg_size; jj++)
					{  // re-evaluate alpha beta and ChiSq
						int i = jj + ii * seg_size;
						int rem_idx = (i + 1) % seg_size;
						if (rem_idx == 0) rem_idx = seg_size;
						int cur_idx = (static_cast<int>(pos_y) - ((seg_size - 1) / 2 + 1) + (i + 1 - rem_idx) / seg_size) * para_config->cam_x + static_cast<int>(pos_x) - ((seg_size - 1) / 2 + 1) + rem_idx - 1;
						float cur_offset = *(offset_map_d + cur_idx);   //calculate map index
						float cur_gain = *(gain_map_d + cur_idx);
						float cur_var = *(var_map_d + cur_idx);
						int x_spl = jj + offset_data_map + xc_int;
						int y_spl = ii + offset_data_map + yc_int;
						int z_spl = spline_z / 2 + 1 + zc_int + LS_stepsize / step_size * (kk - (int)floorf(para_config->num_slice / 2)) - 1;
						int z_spl_exc = spline_z / 2 + 1 + z_exc_int + LS_stepsize / step_size * (kk - (int)floorf(para_config->num_slice / 2)) - 1;  //map z-zc-z_offset to PSFexc template
						if ((x_spl < 0) || (x_spl > spline_x - 1)) continue;
						if ((y_spl < 0) || (y_spl > spline_y - 1)) continue;
						if ((z_spl < 0) || (z_spl > spline_z - 1)) continue;
						if ((z_spl_exc < 0) || (z_spl_exc > spline_z - 1)) continue;
						kernel_DerivativeSpline(offset_fit, &x_spl, &y_spl, &z_spl, &z_spl_exc, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_g, delta_dzg, coef_det_d, coef_exc_d, NewTheta, NewDudt);

						model = NewDudt[3] * NewTheta[3] + NewTheta[4] + cur_var / cur_gain / cur_gain;
						data = (*(data_cur + kk * seg_size * seg_size + ii * seg_size + jj) - cur_offset) / cur_gain + cur_var / cur_gain / cur_gain;
						
						if ((data > 0) && (model > 0))
						{
							NewChiSq += 2 * ((model - data) - data * log(model / data));
							//printf("Chisq increment is %")
						}
						else
						{
							continue;
						}
						for (int k = 0; k < para_config->num_fitting_para; k++) jacobian[k] -= (1 - data / model) * NewDudt[k];
						
						for (int j = 0; j < para_config->num_fitting_para; j++) for (int k = j; k < para_config->num_fitting_para; k++)
						{
							hessian[j * para_config->num_fitting_para + k] += data / model / model * NewDudt[j] * NewDudt[k];
							hessian[k * para_config->num_fitting_para + j] = hessian[j * para_config->num_fitting_para + k];
						}
						
					}
				}
				else
				{
					mu = fmaxf((1 + NewLambda * SCALE_UP) / (1 + NewLambda), 1.3f);
					NewLambda = SCALE_UP * NewLambda;
				}
			}
			if (!offset_fit)
			{
				//printf("Derivative x is %f, y is %f, z is %f, h is %f, bg is %f\n", NewDudt[0], NewDudt[1], NewDudt[2], NewDudt[3], NewDudt[4]);
				*(device_debug_d + idx * iterations * 2 + iter) = NewChiSq;
				*(device_debug_d + idx * iterations * 2 + iter + iterations) = NewDudt[5];
				//printf("current dudt2 is %f\n", NewDudt[2]);
			}
		}  //end to iteration loop
		
		// Calculate CRLB and LogLikelihood
		delete[] Maxjump, jacobian, hessian, NewUpdate, OldUpdate, L, U;
		xc_int = floor(-NewTheta[0]);
		xc_frac = -NewTheta[0] - xc_int;
		yc_int = floor(-NewTheta[1]);
		yc_frac = -NewTheta[1] - yc_int;
		zc_int = floor(-NewTheta[2]);
		zc_frac = -NewTheta[2] - zc_int;
		z_exc_int = floor(-NewTheta[2] - NewTheta[5]);
		z_exc_frac = -NewTheta[2] - NewTheta[5] - z_exc_int;
		kernel_computeDelta3D_exc(z_exc_frac, delta_g, delta_dzg);
		kernel_computeDelta3D_det(xc_frac, yc_frac, zc_frac, delta_f, delta_dxf, delta_dyf, delta_dzf);
		for (int kk = 0; kk < para_config->num_slice; kk++)for (int ii = 0; ii < seg_size; ii++) for (int jj = 0; jj < seg_size; jj++)
		{  // re-evaluate alpha beta
			int i = jj + ii * seg_size;
			int rem_idx = (i + 1) % seg_size;
			if (rem_idx == 0) rem_idx = seg_size;
			int cur_idx = (static_cast<int>(pos_y) - ((seg_size - 1) / 2 + 1) + (i + 1 - rem_idx) / seg_size) * para_config->cam_x + static_cast<int>(pos_x) - ((seg_size - 1) / 2 + 1) + rem_idx - 1;
			float cur_offset = *(offset_map_d + cur_idx);   //calculate map index
			float cur_gain = *(gain_map_d + cur_idx);
			float cur_var = *(var_map_d + cur_idx);
			int x_spl = jj + offset_data_map + xc_int;
			int y_spl = ii + offset_data_map + yc_int;
			int z_spl = spline_z / 2 + 1 + zc_int + LS_stepsize / step_size * (kk - (int)floorf(para_config->num_slice / 2)) - 1;  // -1 explaination: index from 0 in c
			int z_spl_exc = spline_z / 2 + 1 + z_exc_int + LS_stepsize / step_size * (kk - (int)floorf(para_config->num_slice / 2)) - 1;  //map z-zc-z_offset to PSFexc template
			if ((x_spl < 0) || (x_spl > spline_x - 1)) continue;
			if ((y_spl < 0) || (y_spl > spline_y - 1)) continue;
			if ((z_spl < 0) || (z_spl > spline_z - 1)) continue;
			if ((z_spl_exc < 0) || (z_spl_exc > spline_z - 1)) continue;
			kernel_DerivativeSpline(offset_fit, &x_spl, &y_spl, &z_spl, &z_spl_exc, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_g, delta_dzg, coef_det_d, coef_exc_d, NewTheta, NewDudt);
			model = NewDudt[3] * NewTheta[3] + NewTheta[4] + cur_var / cur_gain / cur_gain;
			data = (*(data_cur + kk * seg_size * seg_size + ii * seg_size + jj) - cur_offset) / cur_gain + cur_var / cur_gain / cur_gain;
			for (int j = 0; j < para_config->num_fitting_para; j++) for (int k = j; k < para_config->num_fitting_para; k++)
			{
				M[j * para_config->num_fitting_para + k] += NewDudt[j] * NewDudt[k] / model * 4;
				M[k * para_config->num_fitting_para + j] = M[j * para_config->num_fitting_para + k];
			}
			if ((data > 0) && (model > 0))
			{
				ChiSq_min += 2 * ((model - data) - data * log(model / data));
			}
		}
		if (!offset_fit)
			*(LogLikelihood_d + idx) = ChiSq_min;
		kernel_MatInvN(M, Minv, Diag, para_config->num_fitting_para);
		//if (offset_fit)
		{
			for (int i = 0; i < para_config->num_fitting_para; i++)
			{
				*(fitting_para_d + idx * fit_para_num + i) = NewTheta[i];
				*(CRLBs_d + idx * fit_para_num + i) = Diag[i];
			}
		}
		delete[] NewDudt, M, Minv, Diag;
		return;
	
	}//end to if statement
	
}


extern "C"
void cuda_fitting(dim3 dimgrid, dim3 dimblock, fitting_config* para_config, const float* coef_det_d, const float* coef_exc_d, const float* data_d, const float* offset_map_d, const float* var_map_d,
	const float* gain_map_d, const float* map_ptr_x_d, const float* map_ptr_y_d, float* fitting_para_d, float* CRLBs_d, float* LogLikelihood_d, float* device_debug_d)
{
	kernel_cuda_fitting << <dimgrid, dimblock >> > (para_config, coef_det_d, coef_exc_d, data_d, offset_map_d, var_map_d, gain_map_d, map_ptr_x_d, map_ptr_y_d, fitting_para_d, CRLBs_d, LogLikelihood_d, device_debug_d);
}


 fitting_config::fitting_config(int num_fitting_para_in, int num_launch_thre_in, int num_slice_in, int cam_x_in, int cam_y_in, int fitting_obj_in) 
	{
		num_fitting_para = num_fitting_para_in;
		num_launch_thre = num_launch_thre_in;
		num_slice = num_slice_in;
		cam_x = cam_x_in;
		cam_y = cam_y_in;
		fitting_obj = fitting_obj_in;
	}