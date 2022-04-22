#ifndef GPUSPLINELIB_H
#define GPUSPLINELIB_H

__device__ void kernel_bg_eval(const float* data_cur, const float* offset_map, const float* gain_map, const float* pos_x, const float* pos_y, float* NewTheta);

__device__ void kernel_h_bg_init(const float* data_cur, const float* offset_map, const float* gain_map, const float* pos_x, const float* pos_y, float* NewTheta);

__device__ void kernel_xy_init(const float* data_cur, const float* offset_map, const float* gain_map, const float* pos_x, const float* pos_y, float* NewTheta);

__device__ void kernel_z_init(const float* data_cur, const float* offset_map, const float* gain_map, const float* pos_x, const float* pos_y, float* NewTheta, const float* lat_inten_cali_d);

__device__ void kernel_computeDelta3D_det(float x_delta, float y_delta, float z_delta, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf);

__device__ void kernel_computeDelta3D_exc(float z_exc_frac, float* delta_g, float* delta_dzg);

__device__ void kernel_DerivativeSpline(int* x_spl, int* y_spl, int* z_spl, int* z_spl_exc, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf,float* delta_g,float* delta_dzg, const float* coef_det_d, const float* coef_exc_d, float* theta, float* dudt);

__device__ int kernel_cholesky(float* A, int n, float* L, float* U);

__device__ void kernel_luEvaluate(float* L, float* U, float* b, int n, float* x);

__device__ void kernel_MatInvN(float* M, float* Minv, float* DiagMinv, int sz);

#endif