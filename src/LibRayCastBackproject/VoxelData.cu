// Created by A. Aichert on Wed Dec 21st 2016
#include <iostream>

#include <LibUtilsCuda/UtilsCuda.hxx>

#include <LibUtilsCuda/CudaMemory.h>
#include <LibUtilsCuda/CudaBindlessTexture.h>

#include <LibUtilsCuda/culaut/culaut.hxx>

#include "vr_cuda.hxx"

#include <vector>

#include <NRRD/nrrd_image.hxx>

////////////////////////
// Min/Max binning
////////////////////////

__global__ void kernel_minmax_bin_tex_2_mem(
	int n_binned_x, int n_binned_y, int n_binned_z, cudaTextureObject_t volume_data, float* binned_minmax)
{
	// Find index of current thread
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
	if (idx_x>=n_binned_x) return;
	if (idx_y>=n_binned_y) return;
	if (idx_z>=n_binned_z) return;
	int idx=idx_z*n_binned_x*n_binned_y+ idx_y*n_binned_x+ idx_x;

	// Current (minimal) voxel.
	const float bin_size=2.f;
	float voxel[]={(float)idx_x*bin_size+.5f, (float)idx_y*bin_size+.5f, (float)idx_z*bin_size+.5f};
	float min, max;
	min=max=tex3D<float>(volume_data,voxel[0],voxel[1],voxel[2]);

	// Min/max reduction
	for (float oz=0; oz<bin_size; oz++)
		for (float oy=0; oy<bin_size; oy++)
			for (float ox=0; ox<bin_size; ox++)
			{
				float v=tex3D<float>(volume_data,voxel[0]+ox,voxel[1]+oy,voxel[2]+oz);
				if (v<min) min=v;
				if (v>max) max=v;
			}

	// Write result to binned volume
	binned_minmax[idx*2]  =min;
	binned_minmax[idx*2+1]=max;

}

__global__ void kernel_minmax_bin( int n_x, int n_y, int n_z, float* volume_data, float* binned_minmax)
{
	const int bin_size=2;
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
	
	int out_x = (n_x + 1) / 2;
	int out_y = (n_y + 1) / 2;
	int out_z = (n_z + 1) / 2;
	
	if (idx_x>=out_x) return;
	if (idx_y>=out_y) return;
	if (idx_z>=out_z) return;
	
	int idx=idx_z*out_x*out_y+ idx_y*out_x+ idx_x;

	// Current (minimal) voxel.
	int voxel[]={idx_x*bin_size, idx_y*bin_size, idx_z*bin_size};
	int cx = voxel[0]; if (cx >= n_x) cx = n_x - 1;
	int cy = voxel[1]; if (cy >= n_y) cy = n_y - 1;
	int cz = voxel[2]; if (cz >= n_z) cz = n_z - 1;
	int p = cx + n_x * cy + n_x * n_y * cz;
	float min, max;
	min=volume_data[p*2+0];
	max=volume_data[p*2+1];

	// Min/max reduction
	for (int oz=0; oz<bin_size; oz++)
		for (int oy=0; oy<bin_size; oy++)
			for (int ox=0; ox<bin_size; ox++)
			{
				int tx = voxel[0] + ox; if (tx >= n_x) tx = n_x - 1;
				int ty = voxel[1] + oy; if (ty >= n_y) ty = n_y - 1;
				int tz = voxel[2] + oz; if (tz >= n_z) tz = n_z - 1;
				p = tx + n_x * ty + n_x * n_y * tz;
				float v_min=volume_data[p*2+0];
				float v_max=volume_data[p*2+1];
				if (v_min<min) min=v_min;
				if (v_max>max) max=v_max;
			}

	// Write result to binned volume
	binned_minmax[idx*2]  =min;
	binned_minmax[idx*2+1]=max;

}

// Create a texture with min/max entries of bin_size^3 bins. bin_size must be an even number.
void minmaxBinning(
	int n_x, int n_y, int n_z,
	cudaTextureObject_t volume_data,
	int bin_size,
	UtilsCuda::BindlessTexture3D<float2>** out_texture,
	float & value_min, float& value_max
	)
{
	int n_xb=(int)std::ceil((double)n_x/2);
	int n_yb=(int)std::ceil((double)n_y/2);
	int n_zb=(int)std::ceil((double)n_z/2);
		
	dim3 block_size;
	block_size.x=8;
	block_size.y=8;
	block_size.z=4;
	dim3 grid_size;
	grid_size.x = iDivUp(n_xb,block_size.x);
	grid_size.y = iDivUp(n_yb,block_size.y);
	grid_size.z = iDivUp(n_zb,block_size.z);

	auto* binned_data=new UtilsCuda::MemoryBlock<float>(n_xb*n_yb*n_zb*2);

	kernel_minmax_bin_tex_2_mem<<<grid_size, block_size>>>(n_xb, n_yb, n_zb, volume_data, *binned_data);
	cudaDeviceSynchronize();
	cudaCheckState

	int cur_bin=bin_size/2;
	while (cur_bin>1) {
		cur_bin/=2;
		int src_x = n_xb;
		int src_y = n_yb;
		int src_z = n_zb;
		n_xb=(src_x + 1)/2;
		n_yb=(src_y + 1)/2;
		n_zb=(src_z + 1)/2;
		auto* binned_data2=new UtilsCuda::MemoryBlock<float>(n_xb*n_yb*n_zb*2);
		grid_size.x = iDivUp(n_xb, block_size.x);
		grid_size.y = iDivUp(n_yb, block_size.y);
		grid_size.z = iDivUp(n_zb, block_size.z);
		kernel_minmax_bin<<<grid_size, block_size>>>(src_x, src_y, src_z, *binned_data, *binned_data2);
		cudaCheckState
		delete binned_data;
		binned_data=binned_data2;
	}
	cudaDeviceSynchronize();
	// Final reduction
	int sizes[] = {2,n_xb,n_yb,n_zb};
	NRRD::Image<float> minmax(sizes,4);
	binned_data->readback(minmax);
	value_min=minmax[0];
	value_max=minmax[1];
	for (int i=0;i<n_xb*n_yb*n_zb;i++)
	{
		if (value_min>minmax[i*2+0]) value_min=minmax[i*2+0];
		if (value_max<minmax[i*2+1]) value_max=minmax[i*2+1];
	}



	// Make texture
	*out_texture=new UtilsCuda::BindlessTexture3D<float2>(n_xb,n_yb,n_zb,(float2*)((float*)*binned_data),true,false);
//	delete binned_data;
	cudaCheckState
}
