/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   common_device.cu
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/tinyexr_wrapper.h>

#include <unsupported/Eigen/MatrixFunctions>

#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN


Matrix<float, 3, 4> log_space_lerp(const Matrix<float, 3, 4>& begin, const Matrix<float, 3, 4>& end, float t) {
	Matrix4f A = Matrix4f::Identity();
	A.block<3,4>(0,0) = begin;
	Matrix4f B = Matrix4f::Identity();
	B.block<3,4>(0,0) = end;

	Matrix4f log_space_a_to_b = (B * A.inverse()).log();

	return ((log_space_a_to_b * t).exp() * A).block<3,4>(0,0);
}

GPUMemory<float> load_exr_gpu(const fs::path& path, int* width, int* height) {
	float* out; // width * height * RGBA
	load_exr(&out, width, height, path.str().c_str());
	ScopeGuard mem_guard{[&]() { free(out); }};

	GPUMemory<float> result((*width) * (*height) * 4);
	result.copy_from_host(out);
	return result;
}

GPUMemory<float> load_stbi_gpu(const fs::path& path, int* width, int* height) {
	bool is_hdr = is_hdr_stbi(path);

	void* data; // width * height * RGBA
	int comp;
	if (is_hdr) {
		data = load_stbi_float(path, width, height, &comp, 4);
	} else {
		data = load_stbi(path, width, height, &comp, 4);
	}

	if (!data) {
		throw std::runtime_error{std::string{stbi_failure_reason()}};
	}

	ScopeGuard mem_guard{[&]() { stbi_image_free(data); }};

	if (*width == 0 || *height == 0) {
		throw std::runtime_error{"Image has zero pixels."};
	}

	GPUMemory<float> result((*width) * (*height) * 4);
	if (is_hdr) {
		result.copy_from_host((float*)data);
	} else {
		GPUMemory<uint8_t> bytes((*width) * (*height) * 4);
		bytes.copy_from_host((uint8_t*)data);
		linear_kernel(from_rgba32<float>, 0, nullptr, (*width) * (*height), bytes.data(), result.data(), false, false, 0);
	}

	return result;
}

void save_stbi_gpu(const fs::path& filename, int width, int height, Array4f *gpu_rgba) {
	std::vector<Array4f> cpu_rgba;
	cpu_rgba.resize(width * height);

	CUDA_CHECK_THROW(cudaMemcpy(cpu_rgba.data(), gpu_rgba, width * height * sizeof(Array4f),
		cudaMemcpyDeviceToHost));

	uint8_t* pngpixels = (uint8_t*)malloc(size_t(width) * size_t(height) * 4);
	uint8_t* dst = pngpixels;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			size_t i = x + y*width;
			// *dst++ = (uint8_t)tcnn::clamp(cpu_rgba[i].x() * 255.f, 0.f, 255.f);
			// *dst++ = (uint8_t)tcnn::clamp(cpu_rgba[i].y() * 255.f, 0.f, 255.f);
			// *dst++ = (uint8_t)tcnn::clamp(cpu_rgba[i].z() * 255.f, 0.f, 255.f);
			// *dst++ = (uint8_t)tcnn::clamp(cpu_rgba[i].w() * 255.f, 0.f, 255.f);
			// linear_to_srgb
			*dst++ = (uint8_t)tcnn::clamp(linear_to_srgb(cpu_rgba[i].x()) * 255.f, 0.f, 255.f);
			*dst++ = (uint8_t)tcnn::clamp(linear_to_srgb(cpu_rgba[i].y()) * 255.f, 0.f, 255.f);
			*dst++ = (uint8_t)tcnn::clamp(linear_to_srgb(cpu_rgba[i].z()) * 255.f, 0.f, 255.f);
			*dst++ = (uint8_t)tcnn::clamp(linear_to_srgb(cpu_rgba[i].w()) * 255.f, 0.f, 255.f);
		}
	}

	write_stbi(filename, width, height, 4, pngpixels);
	free(pngpixels);
}

void save_depth_gpu(const fs::path& filename, int width, int height, float *gpu_depth, float depth_scale) {
	uint8_t R, G, B;

	std::vector<float> cpu_depth;
	cpu_depth.resize(width * height);

	CUDA_CHECK_THROW(cudaMemcpy(cpu_depth.data(), gpu_depth, width * height * sizeof(float),
		cudaMemcpyDeviceToHost));

	uint8_t* pngpixels = (uint8_t*)malloc(size_t(width) * size_t(height) * 4);
	uint8_t* dst = pngpixels;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			size_t i = x + y*width;
			depth_rgb(cpu_depth[i] * depth_scale, &R, &G, &B);

			*dst++ = R;	*dst++ = G; *dst++ = B; *dst++ = 255;
		}
	}

	write_stbi(filename, width, height, 4, pngpixels);
	free(pngpixels);
}

NGP_NAMESPACE_END
