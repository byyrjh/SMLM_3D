#define seg_size 19 //unit pixel  has to be odd number
//#define cam_map_size 276  // row
//#define cam_map_size_y 276  // column
#define LS_stepsize 400 //unit nm
//#define SM_seg_size 4000

#define pixel_size_cam 100  //unit nm
#define step_size 10  //unit nm    for psf calibration in z 
#define spline_x 30  //unit has to be even number
#define spline_y 30  //unit
#define spline_z 240  //unit  emitter center (16,16,121)
#define num_coef_per_pix 64  //unit
#define num_coef_per_pix_axial 4  //unit
#define fit_para_num 6  //            x(pixel) y(pixel) z(number of steps in step_size) h bg z_offset
#define init_esti_xy 2 //unit camera pixel
#define init_esti_z 0.5  //unit light sheet step
#define binsize 20  //unit            for bg and h initialization
#define p_value 0.002  // unit    the h value, at which point cdf achieves 0.002, is defined as initial h   smooth out noise induced error 
#define float_zero 0
#define block_size 256
#define iterations 30	//main algorithm para
#define TOLERANCE 1e-6f    //main algorithm para
#define ACCEPTANCE 1.5f    //main algorithm para
#define INIT_LAMBDA 1.0f	//main algorithm para
#define SCALE_UP 5    //main algorithm para
#define SCALE_DOWN 0.2f    //main algorithm para




//application of Bayes theorem to localization 

// localization convention
// origin is at the center of image stack
// from left to right from up to down - to + with respect to center
// from top to bottom of stack - to + with respect to center 
// coordinate x column  y row z stack
// high sampling rate ideal PSF is binned to 31*31*241 followed by normalization to 1, with respect to which the spline coefficients are calculated
// multiplying normalized PSF by 30(AUD), i.e. the brightest ADU is 30, and dividing by 2.2(mean gain) yields 435 photons on the whole image plane
// h value is defined as the brightest pixel in 3 slices of ideal image
// unit of calibrated coefficient data is number of photons

// z initialization calibration to lattice profile is entailed
/*
* stepsize 10nm
* LS_stepsize/step_size=80 data points covers 400 nm (-200 to 200)
*/

