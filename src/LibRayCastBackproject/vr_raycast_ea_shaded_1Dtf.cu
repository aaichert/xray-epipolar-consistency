// Created for EmissionAbsorptionShaded pass
#include <iostream>
#include <LibUtilsCuda/UtilsCuda.hxx>
#include <LibUtilsCuda/CudaMemory.h>
#include <LibUtilsCuda/CudaBindlessTexture.h>
#include <LibUtilsCuda/culaut/culaut.hxx>
#include "vr_cuda.hxx"

__device__ inline float get_sample_clipped(
	cudaTextureObject_t voxel_data,
	float x, float y, float z,
	int n_planes, float* model_C_Pinv_h
	)
{
	float val = tex3D<float>(voxel_data, x, y, z);
	if (n_planes > 0)
	{
		float* E = model_C_Pinv_h + 37;
		if (x * E[0] + y * E[1] + z * E[2] + E[3] < 0.0f)
			val = 0.0f;
	}
	return val;
}

template <int n_c>
__global__ void kernel_raycast_ea_shaded(
	unsigned short n_u, unsigned short n_v,
	float * pixel_data,				//< image data (RGBA)
	float * model_C_Pinv_h,			//< volume model transform (voxels to world)
	cudaTextureObject_t voxel_data,	//< volume data
	float * ray_entry,				//< ray entry
	float * ray_exit,				//< ray exit
	float * noise,					//< noise for ray offsets 
	float   samples_per_voxel,		//< samples per voxel
	cudaTextureObject_t tf			//< transfer function
	)
{
	// Find index of current thread
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx_x>=n_u) return;
	if (idx_y>=n_v) return;
	int idx=idx_y*n_u+idx_x;

	// Access correct pixel
	float *entry=ray_entry+4*idx;
	float *exit =ray_exit+4*idx;
	float *pixel=pixel_data+n_c*idx;

	float ray_length=culaut::xvdistance2<float,3>(entry,exit);
	if (ray_length<=0)
	{
		culaut::xvset<float,n_c>(pixel,0.0f);
		return;
	}

	float step_in_mm=1;
	float4 intensity=make_float4(0,0,0,0);
	float step=1.0/ray_length;
	step/=samples_per_voxel;
	float voxel[3];
	
	// Normalize ray direction for shading
	float ray_direction[3];
	culaut::xvcpy<float,float,3>(ray_direction,exit);
	culaut::xvsub<float,3>(ray_direction,entry);
	culaut::xvnormalize2<float,3>(ray_direction);

	int n_planes = (int)model_C_Pinv_h[36];

	for (float a=noise[idx]*step;a<1.0;a+=step)
	{
		culaut::xvlincomb<float,float,3>(voxel,entry,1.0-a,exit,a);
		float  sample=tex3D<float>(voxel_data,voxel[0],voxel[1],voxel[2]);
		// 1. Calculate lighting/shading
		float gradient[]={
			(get_sample_clipped(voxel_data,voxel[0]+2.0f,voxel[1],voxel[2],n_planes,model_C_Pinv_h)-get_sample_clipped(voxel_data,voxel[0]-2.0f,voxel[1],voxel[2],n_planes,model_C_Pinv_h)),
			(get_sample_clipped(voxel_data,voxel[0],voxel[1]+2.0f,voxel[2],n_planes,model_C_Pinv_h)-get_sample_clipped(voxel_data,voxel[0],voxel[1]-2.0f,voxel[2],n_planes,model_C_Pinv_h)),
			(get_sample_clipped(voxel_data,voxel[0],voxel[1],voxel[2]+2.0f,n_planes,model_C_Pinv_h)-get_sample_clipped(voxel_data,voxel[0],voxel[1],voxel[2]-2.0f,n_planes,model_C_Pinv_h))
		};
		float grad_len = culaut::xvnorm2<float,3>(gradient);
		float lambertian = 1.0f;
		if (grad_len > 1e-4f) {
			culaut::xvnormalize2<float,3>(gradient);
			float dot = culaut::xvdot<float,3>(gradient,ray_direction);
			if (dot < 0.0f) dot = 0.0f;
			if (dot > 1.0f) dot = 1.0f;
			lambertian = 0.35f + 0.65f * dot; // 35% ambient light, 65% diffuse light
		}

		float4 color=tex1D<float4>(tf,sample);
		
		// 2. Apply lambertian shading to RGB components
		color.x *= lambertian;
		color.y *= lambertian;
		color.z *= lambertian;

		color.w/=step_in_mm;
		color.x*=color.w;
		color.y*=color.w;
		color.z*=color.w;
		intensity.x+=color.x*(1.0-intensity.w);
		intensity.y+=color.y*(1.0-intensity.w);
		intensity.z+=color.z*(1.0-intensity.w);
		intensity.w+=color.w*(1.0-intensity.w);
		if (intensity.w>=1.0)
			break;
	}
	if (n_c>=4)
		culaut::xvcpy<float,float,4>(pixel,(float*)&intensity);
	else if (n_c>=3)
		culaut::xvcpy<float,float,3>(pixel,(float*)&intensity);
	else
		culaut::xvset<float,n_c>(pixel,culaut::xvnorm2<float,3>(&intensity.x));
}

void raycast_ea_shaded_1Dtf(
	int n_u, int n_v, int n_c,		//< image size and number of channels
	float * pixel_data_d,			//< image data (RGBA)
	float * model_C_Pinv_d,			//< volume model transform (voxels to world)
	cudaTextureObject_t voxel_data,	//< volume data
	float* ray_entry_d,				//< ray entry
	float* ray_exit_d,				//< ray exit
	float * noise_d,				//< noise for ray offsets 
	float samples_per_voxel,		//< samples per voxel
	cudaTextureObject_t tf			//< transfer function
	)
{
	dim3 block_size;
	block_size.x=16;
	block_size.y=16;
	dim3 grid_size;
	grid_size.x = iDivUp(n_u,block_size.x);
	grid_size.y = iDivUp(n_v,block_size.y);

	if (n_c==1)
		kernel_raycast_ea_shaded<1><<<grid_size, block_size>>>(n_u, n_v, pixel_data_d, model_C_Pinv_d, voxel_data, ray_entry_d, ray_exit_d, noise_d, samples_per_voxel, tf);
	else if (n_c==3)
		kernel_raycast_ea_shaded<3><<<grid_size, block_size>>>(n_u, n_v, pixel_data_d, model_C_Pinv_d, voxel_data, ray_entry_d, ray_exit_d, noise_d, samples_per_voxel, tf);
	else if (n_c==4)
		kernel_raycast_ea_shaded<4><<<grid_size, block_size>>>(n_u, n_v, pixel_data_d, model_C_Pinv_d, voxel_data, ray_entry_d, ray_exit_d, noise_d, samples_per_voxel, tf);
	else std::cerr << __FILE__ << " : " << __LINE__ << " : invalid number of channels!" << std::endl;

	cudaDeviceSynchronize();
	cudaCheckState
}
