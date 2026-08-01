#include "rasterizer_imp.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>



ImageState ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	// obtain(chunk, img.max_contrib, N, 128);
	// obtain(chunk, img.pixel_colors, N, 128);
	// obtain(chunk, img.bucket_count, N, 128);
	// obtain(chunk, img.bucket_offsets, N, 128);
	// cub::DeviceScan::InclusiveSum(nullptr, img.bucket_count_scan_size, img.bucket_count, img.bucket_count, N);
	// obtain(chunk, img.bucket_count_scanning_space, img.bucket_count_scan_size, 128);

    obtain_auto(chunk, img.max_contrib,   N);
    obtain_auto(chunk, img.pixel_colors,  N);
    obtain_auto(chunk, img.bucket_count,  N);
    obtain_auto(chunk, img.bucket_offsets,N);

	cub::DeviceScan::InclusiveSum(nullptr, img.bucket_count_scan_size, img.bucket_count, img.bucket_count, N);
	
	obtain_auto(chunk, img.bucket_count_scanning_space, img.bucket_count_scan_size);

	return img;
}

SampleState SampleState::fromChunk(char *& chunk, size_t C) {
	SampleState sample;
    constexpr int BLOCK_SIZE = 256;
    const size_t P = C * BLOCK_SIZE;

	// obtain(chunk, sample.bucket_to_tile, C, 128);
	// obtain(chunk, sample.bucket_ranges, C, 128);
	// obtain(chunk, sample.T, C * BLOCK_SIZE, 128);
	// obtain(chunk, sample.ar, C * BLOCK_SIZE, 128);
	// obtain(chunk, sample.accum_T, C * BLOCK_SIZE, 128);
	// obtain(chunk, sample.accum_ar, C * BLOCK_SIZE, 128);

	obtain_auto(chunk, sample.bucket_to_tile, C);
	obtain_auto(chunk, sample.bucket_ranges, C);
	obtain_auto(chunk, sample.T, P);
	obtain_auto(chunk, sample.accum_T, P);
	obtain_auto(chunk, sample.ar, P);
	obtain_auto(chunk, sample.accum_ar, P);
	return sample;
}


