#ifndef __fittingpara__
#define __fittingpara__
class fitting_config
{
public:
	int num_launch_thre, num_slice, cam_x, cam_y;
	bool LS_os_fit;
	int fitting_obj; // 0 for FM and 1 for SM
	__host__ __device__ fitting_config(bool LS_os_fit_in, int num_launch_thre_in, int num_slice_in, int cam_x_in, int cam_y_in, int fitting_obj_in);
};
class ls_plane_config
{
public:
	int x_range, y_range, t_range;
	__host__ __device__ ls_plane_config(int x_range, int y_range, int t_range);
};
#endif