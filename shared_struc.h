#ifndef __fittingpara__
#define __fittingpara__
class fitting_config
{
public:
	int num_fitting_para, num_launch_thre, num_slice, cam_x, cam_y;
	int fitting_obj; // 0 for FM and 1 for SM
	__host__ __device__ fitting_config(int num_fitting_para_in, int num_launch_thre_in, int num_slice_in, int cam_x_in, int cam_y_in, int fitting_obj_in);
};
#endif