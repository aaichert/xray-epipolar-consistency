// Created by A. Aichert on Tue Oct 4th 2016
#include <iostream>

#include <LibUtilsCuda/UtilsCuda.hxx>

#include <LibUtilsCuda/CudaMemory.h>
#include <LibUtilsCuda/CudaBindlessTexture.h>

#include <LibUtilsCuda/culaut/culaut.hxx>

#include "vr_cuda.hxx"

template <int n_c>
__global__ void kernel_raycast_drr(
	unsigned short n_u, unsigned short n_v,
	float * __restrict__ pixel_data,				//< image data (RGBA)
	const float * __restrict__ model_vx_to_mm,			//< volume model transform (voxels to world)
	cudaTextureObject_t voxel_data,	//< volume data
	const float* __restrict__ ray_entry,				//< ray entry
	const float* __restrict__ ray_exit,				//< ray exit
	const float * __restrict__ noise,					//< noise for ray offsets
	float   samples_per_voxel		//< samples per voxel
	)
{
	// Find index of current thread
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx_x>=n_u) return;
	if (idx_y>=n_v) return;
	int idx=idx_y*n_u+idx_x;

	// Coalesced 128-bit loads for entry and exit points
	float4 entry_f4 = ((const float4*)ray_entry)[idx];
	float4 exit_f4  = ((const float4*)ray_exit)[idx];

	// Compute ray direction in voxel coordinates
	float dir_x = exit_f4.x - entry_f4.x;
	float dir_y = exit_f4.y - entry_f4.y;
	float dir_z = exit_f4.z - entry_f4.z;

	// Test if we have a valid ray and scale ray_direction to step size
	float ray_length_vx = sqrtf(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
	if (ray_length_vx < 1.0f) {
		if (n_c == 1) {
			pixel_data[idx] = 0.0f;
		} else if (n_c == 2) {
			((float2*)pixel_data)[idx] = make_float2(0.0f, 0.0f);
		} else {
			float* pixel = pixel_data + n_c * idx;
			#pragma unroll
			for (int c = 0; c < n_c; ++c) pixel[c] = 0.0f;
		}
		return;
	}

	// Pre-calculate step scaling factor
	float inv_steps = 1.0f / (ray_length_vx * samples_per_voxel);
	float step_x = dir_x * inv_steps;
	float step_y = dir_y * inv_steps;
	float step_z = dir_z * inv_steps;

	// Go to start position (with random offset)
	float noise_val = noise[idx];
	float jitter = 1.0f - noise_val;
	float voxel_x = entry_f4.x + jitter * step_x;
	float voxel_y = entry_f4.y + jitter * step_y;
	float voxel_z = entry_f4.z + jitter * step_z;

	// Determine the exact number of steps as an integer
	int num_steps = __float2int_rd(ray_length_vx * samples_per_voxel);

	// Accumulate absorption coeffs
	float intensity = 0.0f;
	for (int step = 0; step < num_steps; ++step) {
		float sample = tex3D<float>(voxel_data, voxel_x, voxel_y, voxel_z);
		intensity += sample;
		voxel_x += step_x;
		voxel_y += step_y;
		voxel_z += step_z;
	}

	// How large was one step in mm? Use L1/LDG Cache for the model matrix elements
	// We only need the 3x3 linear part of the transform since translation is ignored
	float step_mm_x = __ldg(&model_vx_to_mm[0]) * step_x + __ldg(&model_vx_to_mm[4]) * step_y + __ldg(&model_vx_to_mm[8]) * step_z;
	float step_mm_y = __ldg(&model_vx_to_mm[1]) * step_x + __ldg(&model_vx_to_mm[5]) * step_y + __ldg(&model_vx_to_mm[9]) * step_z;
	float step_mm_z = __ldg(&model_vx_to_mm[2]) * step_x + __ldg(&model_vx_to_mm[6]) * step_y + __ldg(&model_vx_to_mm[10]) * step_z;
	float mm_per_sample = sqrtf(step_mm_x * step_mm_x + step_mm_y * step_mm_y + step_mm_z * step_mm_z);

	intensity *= mm_per_sample;

	// Coalesced writing based on number of channels
	if (n_c == 1) {
		pixel_data[idx] = intensity;
	} else if (n_c == 2) {
		((float2*)pixel_data)[idx] = make_float2(intensity, ray_length_vx);
	} else if (n_c == 4) {
		((float4*)pixel_data)[idx] = make_float4(intensity, intensity, intensity, intensity);
	} else {
		float* pixel = pixel_data + n_c * idx;
		#pragma unroll
		for (int c = 0; c < n_c; ++c) {
			pixel[c] = intensity;
		}
	}
}

void raycast_drr(
	int n_u, int n_v, int n_c,		//< image size and number of channels
	float * pixel_data_d,			//< image data (RGBA)
	float * model_C_Pinv_d,			//< volume model transform (voxels to world)
	cudaTextureObject_t voxel_data,	//< volume data
	float* ray_entry_d,				//< ray entry
	float* ray_exit_d,				//< ray exit
	float * noise_d,				//< noise for ray offsets 
	float samples_per_voxel		//< samples per voxel
	)
{
	dim3 block_size;
	block_size.x=16;
	block_size.y=16;
	dim3 grid_size;
	grid_size.x = iDivUp(n_u,block_size.x);
	grid_size.y = iDivUp(n_v,block_size.y);

	if (n_c==1)
		kernel_raycast_drr<1><<<grid_size, block_size>>>(n_u, n_v, pixel_data_d, model_C_Pinv_d, voxel_data, ray_entry_d, ray_exit_d, noise_d, samples_per_voxel);
	else if (n_c==2) // special case for backprojection in reconstruction procedures
		kernel_raycast_drr<2><<<grid_size, block_size>>>(n_u, n_v, pixel_data_d, model_C_Pinv_d, voxel_data, ray_entry_d, ray_exit_d, noise_d, samples_per_voxel);
	else if (n_c==3)
		kernel_raycast_drr<3><<<grid_size, block_size>>>(n_u, n_v, pixel_data_d, model_C_Pinv_d, voxel_data, ray_entry_d, ray_exit_d, noise_d, samples_per_voxel);
	else if (n_c==4)
		kernel_raycast_drr<4><<<grid_size, block_size>>>(n_u, n_v, pixel_data_d, model_C_Pinv_d, voxel_data, ray_entry_d, ray_exit_d, noise_d, samples_per_voxel);
	else std::cerr << __FILE__ << " : " << __LINE__ << " : invalid number of channels!" << std::endl;

	cudaDeviceSynchronize();
	cudaCheckState
}
