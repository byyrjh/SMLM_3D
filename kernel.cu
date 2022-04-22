#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 
#include <stdio.h>
#include "GPU_device_fcn.cuh"
#include "para_config.h"


__global__ void kernel_cuda_fitting(const float* coef_det_d, const float* coef_exc_d, const float* data_d, const float* offset_map_d, const float* var_map_d,
	const float* gain_map_d, const float* map_ptr_x_d, const float* map_ptr_y_d, float* fitting_para_d, float* CRLBs_d, float* LogLikelihood_d,float* device_debug_d)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int idx = bx* block_size + tx;
	if (idx >= emitter_num) return;
	if (idx <1000)
	{
		const float pos_x = *(map_ptr_x_d + idx);
		const float pos_y = *(map_ptr_y_d + idx);
		const float* data_cur = data_d + idx * seg_size * seg_size * slice_num;
		float xc_int, yc_int, zc_int, xc_frac, yc_frac, zc_frac;
		float z_exc_int, z_exc_frac;
		float NewTheta[fit_para_num] = { 0 }; //            x y z h bg z_offset
		float OldTheta[fit_para_num] = { 0 }; //            x y z h bg z_offset
		float Maxjump[fit_para_num] = { 1,1,12,9,1.5,10 };  //x y z max value in simu/3*2   h bg max value in simu/2/3*2 
		const int offset_data_map = (spline_x + 1 - seg_size) / 2;
		float jacobian[fit_para_num] = { 0 };
		float hessian[fit_para_num * fit_para_num] = { 0 };
		float delta_f[64] = { 0 }, delta_dxf[64] = { 0 }, delta_dyf[64] = { 0 }, delta_dzf[64] = { 0 };  // mu=h*f*g+bg, f=:psf_det, g=:psf_exc
		float delta_g[4] = { 0 }, delta_dzg[4] = { 0 };
		float NewDudt[fit_para_num] = { 0 }; //dmu/dx  dmu/dy  dmu/dz  dmu/dh  dmu/dbg dmu/dz_offset
		float model, data;
		float NewChiSq;
		float OldChiSq = 1e11;
		float ChiSq_min = 0;
		int errFlag = 0;
		int iter;
		float NewUpdate[fit_para_num] = { 1e11, 1e11, 1e11, 1e11, 1e11, 1e11}, OldUpdate[fit_para_num] = { 1e11, 1e11, 1e11, 1e11, 1e11, 1e11};  //delta x y z h bg z_offset
		float NewLambda = INIT_LAMBDA, OldLambda = INIT_LAMBDA, mu;
		float L[fit_para_num * fit_para_num] = { 0 }, U[fit_para_num * fit_para_num] = { 0 };
		float M[fit_para_num * fit_para_num], Diag[fit_para_num], Minv[fit_para_num * fit_para_num];

		kernel_bg_eval(data_cur, offset_map_d, gain_map_d, &pos_x, &pos_y, NewTheta);
		kernel_h_bg_init(data_cur, offset_map_d, gain_map_d, &pos_x, &pos_y, NewTheta);
		kernel_xy_init(data_cur, offset_map_d, gain_map_d, &pos_x, &pos_y, NewTheta);
		//kernel_z_init(data_cur, offset_map_d, gain_map_d, &pos_x, &pos_y, NewTheta, lat_inten_cali_d);
		*(NewTheta + 2) = 0;
		*(NewTheta + 5) = 0;
		for (int ii = 0; ii < fit_para_num; ii++) OldTheta[ii] = NewTheta[ii];
		xc_int = floor(-NewTheta[0]);
		xc_frac = -NewTheta[0] - xc_int;            // units of xc_int yc_int zc_int z_exc_int are PSF template pixel(lateral) and step(axial) 
		yc_int = floor(-NewTheta[1]);
		yc_frac = -NewTheta[1] - yc_int;
		zc_int = floor(-NewTheta[2]);				// unit of z_offset and z is number of steps in step_size
		zc_frac = -NewTheta[2] - zc_int;
		z_exc_int = floor(-NewTheta[2] - NewTheta[5]);
		z_exc_frac = -NewTheta[2] - NewTheta[5] - z_exc_int;
		kernel_computeDelta3D_det(xc_frac, yc_frac, zc_frac, delta_f, delta_dxf, delta_dyf, delta_dzf);
		kernel_computeDelta3D_exc(z_exc_frac, delta_g, delta_dzg);
		for (int kk = 0; kk < slice_num; kk++)for (int ii = 0; ii < seg_size; ii++) for (int jj = 0; jj < seg_size; jj++) 
		{  // calculate for each pixel. initialize alpha beta and ChiSq
			int i = jj + ii * seg_size;
			int rem_idx = (i + 1) % seg_size;
			if (rem_idx == 0) rem_idx = seg_size;
			int cur_idx = (static_cast<int>(pos_y) - ((seg_size - 1) / 2 + 1) + (i + 1 - rem_idx) / seg_size) * cam_map_size + static_cast<int>(pos_x) - ((seg_size - 1) / 2 + 1) + rem_idx - 1;
			float cur_offset = *(offset_map_d + cur_idx);   //calculate map index
			float cur_gain = *(gain_map_d + cur_idx);
			float cur_var = *(var_map_d + cur_idx);
			int x_spl = jj + offset_data_map + xc_int;
			int y_spl = ii + offset_data_map + yc_int;
			int z_spl = spline_z / 2 + 1 + zc_int + LS_stepsize / step_size * (kk - (int)floorf(slice_num/2))-1;// map z-zc to PSFdet template
			int z_spl_exc = spline_z / 2 + 1 + z_exc_int + LS_stepsize / step_size * (kk - (int)floorf(slice_num / 2))-1;  //map z-zc-z_offset to PSFexc template
			if ((x_spl < 0) || (x_spl > spline_x - 1)) continue;
			if ((y_spl < 0) || (y_spl > spline_y - 1)) continue;
			if ((z_spl < 0) || (z_spl > spline_z - 1)) continue;
			if ((z_spl_exc < 0) || (z_spl_exc > spline_z - 1)) continue;
			kernel_DerivativeSpline(&x_spl, &y_spl, &z_spl, &z_spl_exc, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_g, delta_dzg,  coef_det_d, coef_exc_d, NewTheta, NewDudt);
			
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
			for (int k = 0; k < fit_para_num; k++) jacobian[k] -= (1 - data / model) * NewDudt[k];
				
			for (int j = 0; j < fit_para_num; j++) for (int k = j; k < fit_para_num; k++)
			{
				hessian[j * fit_para_num + k] += data/model/model * NewDudt[j] * NewDudt[k];
				hessian[k * fit_para_num + j] = hessian[j * fit_para_num + k];
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
					for (int i = 0; i < fit_para_num; i++) 
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
					//printf("current chisq is %f \n", NewChiSq);
				}

				for (int i = 0; i < fit_para_num; i++)
				{
					hessian[i * fit_para_num + i] = hessian[i * fit_para_num + i] * mu;
				}
				memset(L, 0, fit_para_num * fit_para_num * sizeof(float));
				memset(U, 0, fit_para_num * fit_para_num * sizeof(float));
				errFlag = kernel_cholesky(hessian, fit_para_num, L, U);
				if (errFlag == 0) 
				{
					for (int i = 0; i < fit_para_num; i++) 
					{
						OldTheta[i] = NewTheta[i];
						OldUpdate[i] = NewUpdate[i];
					}
					OldLambda = NewLambda;
					OldChiSq = NewChiSq;
					kernel_luEvaluate(L, U, jacobian, fit_para_num, NewUpdate);
					//updateFitParameters
					for (int ll = 0; ll < fit_para_num; ll++) 
					{
						if (NewUpdate[ll] / OldUpdate[ll] < -0.5f) Maxjump[ll] = Maxjump[ll] * 0.5;
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
					memset(jacobian, 0, fit_para_num * sizeof(float));
					memset(hessian, 0, fit_para_num * fit_para_num * sizeof(float));
					kernel_computeDelta3D_exc(z_exc_frac, delta_g, delta_dzg);
					kernel_computeDelta3D_det(xc_frac, yc_frac, zc_frac, delta_f, delta_dxf, delta_dyf, delta_dzf);
					for (int kk = 0; kk < slice_num; kk++)for (int ii = 0; ii < seg_size; ii++) for (int jj = 0; jj < seg_size; jj++)
					{  // re-evaluate alpha beta and ChiSq
						int i = jj + ii * seg_size;
						int rem_idx = (i + 1) % seg_size;
						if (rem_idx == 0) rem_idx = seg_size;
						int cur_idx = (static_cast<int>(pos_y) - ((seg_size - 1) / 2 + 1) + (i + 1 - rem_idx) / seg_size) * cam_map_size + static_cast<int>(pos_x) - ((seg_size - 1) / 2 + 1) + rem_idx - 1;
						float cur_offset = *(offset_map_d + cur_idx);   //calculate map index
						float cur_gain = *(gain_map_d + cur_idx);
						float cur_var = *(var_map_d + cur_idx);
						int x_spl = jj + offset_data_map + xc_int;
						int y_spl = ii + offset_data_map + yc_int;
						int z_spl = spline_z / 2 + 1 + zc_int + LS_stepsize / step_size * (kk - (int)floorf(slice_num / 2))-1;
						int z_spl_exc = spline_z / 2 + 1 + z_exc_int + LS_stepsize / step_size * (kk - (int)floorf(slice_num / 2))-1;  //map z-zc-z_offset to PSFexc template
						if ((x_spl < 0) || (x_spl > spline_x - 1)) continue;
						if ((y_spl < 0) || (y_spl > spline_y - 1)) continue;
						if ((z_spl < 0) || (z_spl > spline_z - 1)) continue;
						if ((z_spl_exc < 0) || (z_spl_exc > spline_z - 1)) continue;
						kernel_DerivativeSpline(&x_spl, &y_spl, &z_spl, &z_spl_exc, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_g, delta_dzg, coef_det_d, coef_exc_d, NewTheta, NewDudt);

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
						for (int k = 0; k < fit_para_num; k++) jacobian[k] -= (1 - data / model) * NewDudt[k];

						for (int j = 0; j < fit_para_num; j++) for (int k = j; k < fit_para_num; k++)
						{
							hessian[j * fit_para_num + k] += data / model / model * NewDudt[j] * NewDudt[k];
							hessian[k * fit_para_num + j] = hessian[j * fit_para_num + k];
						}
					}	
				}
				else
				{
					mu = fmaxf((1 + NewLambda * SCALE_UP) / (1 + NewLambda), 1.3f);
					NewLambda = SCALE_UP * NewLambda;
				}
			}
			*(device_debug_d + idx * 100 + iter) = NewDudt[0];
			*(device_debug_d + idx * 100 + iter + 50) = NewDudt[5];

		}  //end to iteration loop
		
		// Calculating the CRLB and LogLikelihood
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
		pix_count = 0;
		for (int kk = 0; kk < slice_num; kk++)for (int ii = 0; ii < seg_size; ii++) for (int jj = 0; jj < seg_size; jj++)
		{  // re-evaluate alpha beta
			int i = jj + ii * seg_size;
			int rem_idx = (i + 1) % seg_size;
			if (rem_idx == 0) rem_idx = seg_size;
			int cur_idx = (static_cast<int>(pos_y) - ((seg_size - 1) / 2 + 1) + (i + 1 - rem_idx) / seg_size) * cam_map_size + static_cast<int>(pos_x) - ((seg_size - 1) / 2 + 1) + rem_idx - 1;
			float cur_offset = *(offset_map_d + cur_idx);   //calculate map index
			float cur_gain = *(gain_map_d + cur_idx);
			float cur_var = *(var_map_d + cur_idx);
			int x_spl = jj + offset_data_map + xc_int;
			int y_spl = ii + offset_data_map + yc_int;
			int z_spl = spline_z / 2 + 1 + zc_int + LS_stepsize / step_size * (kk - (int)floorf(slice_num / 2))-1;  // -1 explaination: index from 0 in c
			int z_spl_exc = spline_z / 2 + 1 + z_exc_int + LS_stepsize / step_size * (kk - (int)floorf(slice_num / 2))-1;  //map z-zc-z_offset to PSFexc template
			if ((x_spl < 0) || (x_spl > spline_x - 1)) continue;
			if ((y_spl < 0) || (y_spl > spline_y - 1)) continue;
			if ((z_spl < 0) || (z_spl > spline_z - 1)) continue;
			if ((z_spl_exc < 0) || (z_spl_exc > spline_z - 1)) continue;
			kernel_DerivativeSpline(&x_spl, &y_spl, &z_spl, &z_spl_exc, delta_f, delta_dxf, delta_dyf, delta_dzf, delta_g, delta_dzg, coef_det_d, coef_exc_d, NewTheta, NewDudt);
			model = NewDudt[3] * NewTheta[3] + NewTheta[4] + cur_var / cur_gain / cur_gain;
			data = (*(data_cur + kk * seg_size * seg_size + ii * seg_size + jj) - cur_offset) / cur_gain + cur_var / cur_gain / cur_gain;
			for (int j = 0; j < fit_para_num; j++) for (int k = j; k < fit_para_num; k++) 
			{
				M[j * fit_para_num + k] += NewDudt[j] * NewDudt[k] / model * 4;
				M[k * fit_para_num + j] = M[j * fit_para_num + k];
			}
			if ((data > 0) && (model > 0))
			{
				pix_count++;
				ChiSq_min += 2 * ((model - data) - data * log(model / data));
			}	
		}
		*(LogLikelihood_d + idx) = ChiSq_min;
		kernel_MatInvN(M, Minv, Diag, fit_para_num);

		for (int i = 0; i < fit_para_num; i++)
		{
			*(fitting_para_d + idx * fit_para_num + i) = NewTheta[i];
			*(CRLBs_d + idx * fit_para_num + i) = Diag[i];
		}
		return;
		
	}//end to if statement
	
}


extern "C"
void cuda_fitting(dim3 dimgrid, dim3 dimblock, const float* coef_det_d, const float* coef_exc_d, const float* data_d, const float* offset_map_d, const float* var_map_d,
	const float* gain_map_d, const float* map_ptr_x_d, const float* map_ptr_y_d, float* fitting_para_d, float* CRLBs_d, float* LogLikelihood_d, float* device_debug_d)
{
	kernel_cuda_fitting <<<dimgrid, dimblock >>> (coef_det_d, coef_exc_d, data_d, offset_map_d, var_map_d, gain_map_d, map_ptr_x_d, map_ptr_y_d, fitting_para_d, CRLBs_d, LogLikelihood_d, device_debug_d);
}
