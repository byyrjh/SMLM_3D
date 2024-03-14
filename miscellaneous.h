#include <vector>
//#include <algorithm>
using std::vector;
/*
typedef std::vector<double> DoubleVec;

int findNearestNeighbourIndex(const double ac_dfValue, DoubleVec x)
{
    double lv_dfDistance = DBL_MAX;
    int lv_nIndex = -1;

    for (unsigned int i = 0; i < x.size(); i++) {
        double newDist = ac_dfValue - x[i];
        if (newDist >= 0 && newDist < lv_dfDistance) {
            lv_dfDistance = newDist;
            lv_nIndex = i;
        }
    }

    return lv_nIndex;
}

DoubleVec interpolation(DoubleVec x, DoubleVec y,
    DoubleVec xx)
{
    double dx, dy;
    DoubleVec slope, intercept, result;
    slope.resize(x.size());
    intercept.resize(x.size());
    result.resize(xx.size());
    int indiceEnVector;

    for (unsigned i = 0; i < x.size(); i++) {
        if (i < x.size() - 1) {
            dx = x[i + 1] - x[i];
            dy = y[i + 1] - y[i];
            slope[i] = dy / dx;
            intercept[i] = y[i] - x[i] * slope[i];
        }
        else {
            slope[i] = slope[i - 1];
            intercept[i] = intercept[i - 1];
        }
    }

    for (unsigned i = 0; i < xx.size(); i++) {
        indiceEnVector = findNearestNeighbourIndex(xx[i], x);
        if (indiceEnVector != -1) {
            result[i] = slope[indiceEnVector] *
                xx[i] + intercept[indiceEnVector];
        }
        else
            result[i] = DBL_MAX;
    }
    return result;
}
*/
#ifndef FUNCTIONS_H_INCLUDED
#define FUNCTIONS_H_INCLUDED
vector<vector<vector<float>>> LS_os_calc_SM(float* fitting_para_SM_h, float* CRLBs_SM_h, float* map_ptr_t_h_SM, float* map_ptr_x_h_SM, float* map_ptr_y_h_SM, float cam_map_size_ptr, float cam_map_size_y_ptr, float num_vol_ptr, float SM_num, float xybinsize_SM, int smooth_seg_SM);
vector<vector<vector<vector<float>>>> LS_os_calc_SM_full(float* fitting_para_SM_h, float* CRLBs_SM_h, float* map_ptr_t_h_SM, float* map_ptr_x_h_SM, float* map_ptr_y_h_SM, float* map_ptr_z_h_SM, float cam_map_size_ptr, float cam_map_size_y_ptr, float num_vol_ptr, float SM_num, float xybinsize_SM, int smooth_seg_SM);
void LS_os_calc_FM(float* fitting_para_h, double* test_LS_os, int& num_vol, float& stationary_pos_ptr, float& vol_per_hyper_ptr, int& smooth_seg, int& FM_trace);
void LS_plane_fitting(float* LS_plane_coeff, float* LS_os_data, int x, int y, int t);
void MatInvN(float* M, float* Minv, int sz);
void LS_plane_coeff_smooth(float* LS_plane_coeff_raw, float* LS_plane_coeff_smo, int smooth_range, int t_range);
void Smooth_generic(float* output, float* input, int smooth_range);
void LS_os_filter(float* output, float* input, int x, int y, int t);
vector<vector<vector<vector<float>>>> LS_os_filter_full(vector<vector<vector<vector<float>>>>* LS_os_data_raw, int x, int y, int z, int t);
#endif