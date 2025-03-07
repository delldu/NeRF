/*
* Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

/** @file   common.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Shared functionality among multiple neural-graphics-primitives components.
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/random_val.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <Eigen/Dense>

NGP_NAMESPACE_BEGIN

using precision_t = tcnn::network_precision_t;


// The maximum depth that can be produced when rendering a frame.
// Chosen somewhat low (rather than std::numeric_limits<float>::infinity())
// to permit numerically stable reprojection and DLSS operation,
// even when rendering the infinitely distant horizon.
inline constexpr __device__ float MAX_DEPTH() { return 16384.0f; }

template <typename T>
class Buffer2D {
public:
	Buffer2D() = default;
	Buffer2D(const Eigen::Vector2i& resolution) {
		resize(resolution);
	}

	T* data() const {
		return m_data.data();
	}

	size_t bytes() const {
		return m_data.bytes();
	}

	void resize(const Eigen::Vector2i& resolution) {
		m_data.resize(resolution.prod());
		m_resolution = resolution;
	}

	const Eigen::Vector2i& resolution() const {
		return m_resolution;
	}

	Buffer2DView<T> view() const {
		// Row major for now.
		return {data(), m_resolution};
	}

	Buffer2DView<const T> const_view() const {
		// Row major for now.
		return {data(), m_resolution};
	}

private:
	tcnn::GPUMemory<T> m_data;
	Eigen::Vector2i m_resolution;
};

inline NGP_HOST_DEVICE float srgb_to_linear(float srgb) {
	if (srgb <= 0.04045f) {
		return srgb / 12.92f;
	} else {
		return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
	}
}

inline NGP_HOST_DEVICE Eigen::Array3f srgb_to_linear(const Eigen::Array3f& x) {
	return {srgb_to_linear(x.x()), srgb_to_linear(x.y()), (srgb_to_linear(x.z()))};
}

inline NGP_HOST_DEVICE float srgb_to_linear_derivative(float srgb) {
	if (srgb <= 0.04045f) {
		return 1.0f / 12.92f;
	} else {
		return 2.4f / 1.055f * std::pow((srgb + 0.055f) / 1.055f, 1.4f);
	}
}

inline NGP_HOST_DEVICE Eigen::Array3f srgb_to_linear_derivative(const Eigen::Array3f& x) {
	return {srgb_to_linear_derivative(x.x()), srgb_to_linear_derivative(x.y()), (srgb_to_linear_derivative(x.z()))};
}

inline NGP_HOST_DEVICE float linear_to_srgb(float linear) {
	if (linear < 0.0031308f) {
		return 12.92f * linear;
	} else {
		return 1.055f * std::pow(linear, 0.41666f) - 0.055f;
	}
}

inline NGP_HOST_DEVICE Eigen::Array3f linear_to_srgb(const Eigen::Array3f& x) {
	return {linear_to_srgb(x.x()), linear_to_srgb(x.y()), (linear_to_srgb(x.z()))};
}

inline NGP_HOST_DEVICE float linear_to_srgb_derivative(float linear) {
	if (linear < 0.0031308f) {
		return 12.92f;
	} else {
		return 1.055f * 0.41666f * std::pow(linear, 0.41666f - 1.0f);
	}
}

inline NGP_HOST_DEVICE Eigen::Array3f linear_to_srgb_derivative(const Eigen::Array3f& x) {
	return {linear_to_srgb_derivative(x.x()), linear_to_srgb_derivative(x.y()), (linear_to_srgb_derivative(x.z()))};
}

template <uint32_t N_DIMS, typename T>
__device__ void deposit_image_gradient(const Eigen::Matrix<float, N_DIMS, 1>& value, T* __restrict__ gradient, T* __restrict__ gradient_weight, const Eigen::Vector2i& resolution, const Eigen::Vector2f& pos) {
	const Eigen::Vector2f pos_float = resolution.cast<float>().cwiseProduct(pos);
	const Eigen::Vector2i texel = pos_float.cast<int>();

	const Eigen::Vector2f weight = pos_float - texel.cast<float>();

	auto deposit_val = [&](const Eigen::Matrix<float, N_DIMS, 1>& value, T weight, Eigen::Vector2i pos) {
		pos.x() = std::max(std::min(pos.x(), resolution.x()-1), 0);
		pos.y() = std::max(std::min(pos.y(), resolution.y()-1), 0);

#if TCNN_MIN_GPU_ARCH >= 60 // atomicAdd(__half2) is only supported with compute capability 60 and above
		if (std::is_same<T, __half>::value) {
			for (uint32_t c = 0; c < N_DIMS; c += 2) {
				atomicAdd((__half2*)&gradient[(pos.x() + pos.y() * resolution.x()) * N_DIMS + c], {(T)value[c] * weight, (T)value[c+1] * weight});
				atomicAdd((__half2*)&gradient_weight[(pos.x() + pos.y() * resolution.x()) * N_DIMS + c], {weight, weight});
			}
		} else
#endif
		{
			for (uint32_t c = 0; c < N_DIMS; ++c) {
				atomicAdd(&gradient[(pos.x() + pos.y() * resolution.x()) * N_DIMS + c], (T)value[c] * weight);
				atomicAdd(&gradient_weight[(pos.x() + pos.y() * resolution.x()) * N_DIMS + c], weight);
			}
		}
	};

	deposit_val(value, (1 - weight.x()) * (1 - weight.y()), {texel.x(), texel.y()});
	deposit_val(value, (weight.x()) * (1 - weight.y()), {texel.x()+1, texel.y()});
	deposit_val(value, (1 - weight.x()) * (weight.y()), {texel.x(), texel.y()+1});
	deposit_val(value, (weight.x()) * (weight.y()), {texel.x()+1, texel.y()+1});
}

struct FoveationPiecewiseQuadratic {
	NGP_HOST_DEVICE FoveationPiecewiseQuadratic() = default;

	FoveationPiecewiseQuadratic(float center_pixel_steepness, float center_inverse_piecewise_y, float center_radius) {
		float center_inverse_radius = center_radius * center_pixel_steepness;
		float left_inverse_piecewise_switch = center_inverse_piecewise_y - center_inverse_radius;
		float right_inverse_piecewise_switch = center_inverse_piecewise_y + center_inverse_radius;

		if (left_inverse_piecewise_switch < 0) {
			left_inverse_piecewise_switch = 0.0f;
		}

		if (right_inverse_piecewise_switch > 1) {
			right_inverse_piecewise_switch = 1.0f;
		}

		float am = center_pixel_steepness;
		float d = (right_inverse_piecewise_switch - left_inverse_piecewise_switch) / center_pixel_steepness / 2;

		// binary search for l,r,bm since analytical is very complex
		float bm;
		float m_min = 0.0f;
		float m_max = 1.0f;
		for (uint32_t i = 0; i < 20; i++) {
			float m = (m_min + m_max) / 2.0f;
			float l = m - d;
			float r = m + d;

			bm = -((am - 1) * l * l) / (r * r - 2 * r + l * l + 1);

			float l_actual = (left_inverse_piecewise_switch - bm) / am;
			float r_actual = (right_inverse_piecewise_switch - bm) / am;
			float m_actual = (l_actual + r_actual) / 2;

			if (m_actual > m) {
				m_min = m;
			} else {
				m_max = m;
			}
		}

		float l = (left_inverse_piecewise_switch - bm) / am;
		float r = (right_inverse_piecewise_switch - bm) / am;

		// Full linear case. Default construction covers this.
		if ((l == 0.0f && r == 1.0f) || (am == 1.0f)) {
			return;
		}

		// write out solution
		switch_left = l;
		switch_right = r;
		this->am = am;
		al = (am - 1) / (r * r - 2 * r + l * l + 1);
		bl = (am * (r * r - 2 * r + 1) + am * l * l + (2 - 2 * am) * l) / (r * r - 2 * r + l * l + 1);
		cl = 0;
		this->bm = bm = -((am - 1) * l * l) / (r * r - 2 * r + l * l + 1);
		ar = -(am - 1) / (r * r - 2 * r + l * l + 1);
		br = (am * (r * r + 1) - 2 * r + am * l * l) / (r * r - 2 * r + l * l + 1);
		cr = -(am * r * r - r * r + (am - 1) * l * l) / (r * r - 2 * r + l * l + 1);

		inv_switch_left = am * switch_left + bm;
		inv_switch_right = am * switch_right + bm;
	}

	// left parabola: al * x^2 + bl * x + cl
	float al = 0.0f, bl = 0.0f, cl = 0.0f;
	// middle linear piece: am * x + bm.  am should give 1:1 pixel mapping between warped size and full size.
	float am = 1.0f, bm = 0.0f;
	// right parabola: al * x^2 + bl * x + cl
	float ar = 0.0f, br = 0.0f, cr = 0.0f;

	// points where left and right switch over from quadratic to linear
	float switch_left = 0.0f, switch_right = 1.0f;
	// same, in inverted space
	float inv_switch_left = 0.0f, inv_switch_right = 1.0f;

	NGP_HOST_DEVICE float warp(float x) const {
		x = tcnn::clamp(x, 0.0f, 1.0f);
		if (x < switch_left) {
			return al * x * x + bl * x + cl;
		} else if (x > switch_right) {
			return ar * x * x + br * x + cr;
		} else {
			return am * x + bm;
		}
	}

	NGP_HOST_DEVICE float unwarp(float y) const {
		y = tcnn::clamp(y, 0.0f, 1.0f);
		if (y < inv_switch_left) {
			return (std::sqrt(-4 * al * cl + 4 * al * y + bl * bl) - bl) / (2 * al);
		} else if (y > inv_switch_right) {
			return (std::sqrt(-4 * ar * cr + 4 * ar * y + br * br) - br) / (2 * ar);
		} else {
			return (y - bm) / am;
		}
	}

	NGP_HOST_DEVICE float density(float x) const {
		x = tcnn::clamp(x, 0.0f, 1.0f);
		if (x < switch_left) {
			return 2 * al * x + bl;
		} else if (x > switch_right) {
			return 2 * ar * x + br;
		} else {
			return am;
		}
	}
};

struct Foveation {
	NGP_HOST_DEVICE Foveation() = default;

	Foveation(const Eigen::Vector2f& center_pixel_steepness, const Eigen::Vector2f& center_inverse_piecewise_y, const Eigen::Vector2f& center_radius)
	: warp_x{center_pixel_steepness.x(), center_inverse_piecewise_y.x(), center_radius.x()}, warp_y{center_pixel_steepness.y(), center_inverse_piecewise_y.y(), center_radius.y()} {}

	FoveationPiecewiseQuadratic warp_x, warp_y;

	NGP_HOST_DEVICE Eigen::Vector2f warp(const Eigen::Vector2f& x) const {
		return {warp_x.warp(x.x()), warp_y.warp(x.y())};
	}

	NGP_HOST_DEVICE Eigen::Vector2f unwarp(const Eigen::Vector2f& y) const {
		return {warp_x.unwarp(y.x()), warp_y.unwarp(y.y())};
	}

	NGP_HOST_DEVICE float density(const Eigen::Vector2f& x) const {
		return warp_x.density(x.x()) * warp_y.density(x.y());
	}
};

template <typename T>
NGP_HOST_DEVICE inline void opencv_lens_distortion_delta(const T* extra_params, const T u, const T v, T* du, T* dv) {
	const T k1 = extra_params[0];
	const T k2 = extra_params[1];
	const T p1 = extra_params[2];
	const T p2 = extra_params[3];

	const T u2 = u * u;
	const T uv = u * v;
	const T v2 = v * v;
	const T r2 = u2 + v2;
	const T radial = k1 * r2 + k2 * r2 * r2;
	*du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
	*dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
}

template <typename T>
NGP_HOST_DEVICE inline void opencv_fisheye_lens_distortion_delta(const T* extra_params, const T u, const T v, T* du, T* dv) {
	const T k1 = extra_params[0];
	const T k2 = extra_params[1];
	const T k3 = extra_params[2];
	const T k4 = extra_params[3];

	const T r = std::sqrt(u * u + v * v);

	if (r > T(std::numeric_limits<double>::epsilon())) {
		const T theta = std::atan(r);
		const T theta2 = theta * theta;
		const T theta4 = theta2 * theta2;
		const T theta6 = theta4 * theta2;
		const T theta8 = theta4 * theta4;
		const T thetad =
			theta * (T(1) + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
		*du = u * thetad / r - u;
		*dv = v * thetad / r - v;
	} else {
		*du = T(0);
		*dv = T(0);
	}
}

template <typename T, typename F>
NGP_HOST_DEVICE inline void iterative_lens_undistortion(const T* params, T* u, T* v, F distortion_fun) {
	// Parameters for Newton iteration using numerical differentiation with
	// central differences, 100 iterations should be enough even for complex
	// camera models with higher order terms.
	const uint32_t kNumIterations = 100;
	const float kMaxStepNorm = 1e-10f;
	const float kRelStepSize = 1e-6f;

	Eigen::Matrix2f J;
	const Eigen::Vector2f x0(*u, *v);
	Eigen::Vector2f x(*u, *v);
	Eigen::Vector2f dx;
	Eigen::Vector2f dx_0b;
	Eigen::Vector2f dx_0f;
	Eigen::Vector2f dx_1b;
	Eigen::Vector2f dx_1f;

	for (uint32_t i = 0; i < kNumIterations; ++i) {
		const float step0 = std::max(std::numeric_limits<float>::epsilon(), std::abs(kRelStepSize * x(0)));
		const float step1 = std::max(std::numeric_limits<float>::epsilon(), std::abs(kRelStepSize * x(1)));
		distortion_fun(params, x(0), x(1), &dx(0), &dx(1));
		distortion_fun(params, x(0) - step0, x(1), &dx_0b(0), &dx_0b(1));
		distortion_fun(params, x(0) + step0, x(1), &dx_0f(0), &dx_0f(1));
		distortion_fun(params, x(0), x(1) - step1, &dx_1b(0), &dx_1b(1));
		distortion_fun(params, x(0), x(1) + step1, &dx_1f(0), &dx_1f(1));
		J(0, 0) = 1 + (dx_0f(0) - dx_0b(0)) / (2 * step0);
		J(0, 1) = (dx_1f(0) - dx_1b(0)) / (2 * step1);
		J(1, 0) = (dx_0f(1) - dx_0b(1)) / (2 * step0);
		J(1, 1) = 1 + (dx_1f(1) - dx_1b(1)) / (2 * step1);
		const Eigen::Vector2f step_x = J.inverse() * (x + dx - x0);
		x -= step_x;
		if (step_x.squaredNorm() < kMaxStepNorm) {
			break;
		}
	}

	*u = x(0);
	*v = x(1);
}

template <typename T>
NGP_HOST_DEVICE inline void iterative_opencv_lens_undistortion(const T* params, T* u, T* v) {
	iterative_lens_undistortion(params, u, v, opencv_lens_distortion_delta<T>);
}

template <typename T>
NGP_HOST_DEVICE inline void iterative_opencv_fisheye_lens_undistortion(const T* params, T* u, T* v) {
	iterative_lens_undistortion(params, u, v, opencv_fisheye_lens_distortion_delta<T>);
}

inline NGP_HOST_DEVICE Ray pixel_to_ray_pinhole(
	uint32_t spp,
	const Eigen::Vector2i& pixel,
	const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
	const Eigen::Vector2f& screen_center
) {
	const Eigen::Vector2f uv = pixel.cast<float>().cwiseQuotient(resolution.cast<float>());

	Eigen::Vector3f dir = {
		(uv.x() - screen_center.x()) * (float)resolution.x() / focal_length.x(),
		(uv.y() - screen_center.y()) * (float)resolution.y() / focal_length.y(),
		1.0f
	};

	dir = camera_matrix.block<3, 3>(0, 0) * dir;

	Eigen::Vector3f origin = camera_matrix.col(3);
	return {origin, dir};
}

inline NGP_HOST_DEVICE Eigen::Matrix<float, 3, 4> get_xform_given_rolling_shutter(const TrainingXForm& training_xform, const Eigen::Vector4f& rolling_shutter, const Eigen::Vector2f& uv, float motionblur_time) {
	float pixel_t = rolling_shutter.x() + rolling_shutter.y() * uv.x() + rolling_shutter.z() * uv.y() + rolling_shutter.w() * motionblur_time;

	Eigen::Vector3f pos = training_xform.start.col(3) + (training_xform.end.col(3) - training_xform.start.col(3)) * pixel_t;
	Eigen::Quaternionf rot = Eigen::Quaternionf(training_xform.start.block<3, 3>(0, 0)).slerp(pixel_t, Eigen::Quaternionf(training_xform.end.block<3, 3>(0, 0)));

	Eigen::Matrix<float, 3, 4> rv;
	rv.col(3) = pos;
	rv.block<3, 3>(0, 0) = Eigen::Quaternionf(rot).normalized().toRotationMatrix();
	return rv;
}

inline NGP_HOST_DEVICE Eigen::Vector3f f_theta_undistortion(const Eigen::Vector2f& uv, const float* params, const Eigen::Vector3f& error_direction) {
	// we take f_theta intrinsics to be: r0, r1, r2, r3, resx, resy; we rescale to whatever res the intrinsics specify.
	float xpix = uv.x() * params[5];
	float ypix = uv.y() * params[6];
	float norm = sqrtf(xpix*xpix + ypix*ypix);
	float alpha = params[0] + norm * (params[1] + norm * (params[2] + norm * (params[3] + norm * params[4])));
	float sin_alpha, cos_alpha;
	sincosf(alpha, &sin_alpha, &cos_alpha);
	if (cos_alpha <= std::numeric_limits<float>::min() || norm == 0.f) {
		return error_direction;
	}
	sin_alpha *= 1.f / norm;
	return { sin_alpha * xpix, sin_alpha * ypix, cos_alpha };
}

inline NGP_HOST_DEVICE Eigen::Vector3f latlong_to_dir(const Eigen::Vector2f& uv) {
	float theta = (uv.y() - 0.5f) * PI();
	float phi = (uv.x() - 0.5f) * PI() * 2.0f;
	float sp, cp, st, ct;
	sincosf(theta, &st, &ct);
	sincosf(phi, &sp, &cp);
	return {sp * ct, st, cp * ct};
}

inline NGP_HOST_DEVICE Eigen::Vector3f equirectangular_to_dir(const Eigen::Vector2f& uv) {
	float ct = (uv.y() - 0.5f) * 2.0f;
	float st = std::sqrt(std::max(1.0f - ct * ct, 0.0f));
	float phi = (uv.x() - 0.5f) * PI() * 2.0f;
	float sp, cp;
	sincosf(phi, &sp, &cp);
	return {sp * st, ct, cp * st};
}

inline NGP_HOST_DEVICE Ray uv_to_ray(
	uint32_t spp,
	const Eigen::Vector2f& uv,
	const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
	const Eigen::Vector2f& screen_center,
	const Eigen::Vector3f& parallax_shift = Eigen::Vector3f::Zero(),
	float near_distance = 0.0f,
	float focus_z = 1.0f, // plane_z -- 1.36364
	float aperture_size = 0.0f,
	const Foveation& foveation = {},
	Buffer2DView<const uint8_t> hidden_area_mask = {},
	const Lens& lens = {},
	Buffer2DView<const Eigen::Vector2f> distortion = {}
) {
	Eigen::Vector2f warped_uv = foveation.warp(uv);

	// Check the hidden area mask _after_ applying foveation, because foveation will be undone
	// before blitting to the framebuffer to which the hidden area mask corresponds.
	if (hidden_area_mask && !hidden_area_mask.at(warped_uv)) {
		return Ray::invalid();
	}

	Eigen::Vector3f dir;
	if (lens.mode == ELensMode::FTheta) {
		dir = f_theta_undistortion(warped_uv - screen_center, lens.params, {0.f, 0.f, 0.f});
		if (dir == Eigen::Vector3f::Zero()) {
			return Ray::invalid();
		}
	} else if (lens.mode == ELensMode::LatLong) {
		dir = latlong_to_dir(warped_uv);
	} else if (lens.mode == ELensMode::Equirectangular) {
		dir = equirectangular_to_dir(warped_uv);
	} else {
		dir = {
			(warped_uv.x() - screen_center.x()) * (float)resolution.x() / focal_length.x(),
			(warped_uv.y() - screen_center.y()) * (float)resolution.y() / focal_length.y(),
			1.0f
		};

		if (lens.mode == ELensMode::OpenCV) {
			iterative_opencv_lens_undistortion(lens.params, &dir.x(), &dir.y());
		} else if (lens.mode == ELensMode::OpenCVFisheye) {
			iterative_opencv_fisheye_lens_undistortion(lens.params, &dir.x(), &dir.y());
		}
	}

	if (distortion) {
		dir.head<2>() += distortion.at_lerp(warped_uv);
	}

	Eigen::Vector3f head_pos = {parallax_shift.x(), parallax_shift.y(), 0.f};
	dir -= head_pos * parallax_shift.z(); // we could use focus_z here in the denominator. for now, we pack m_scale in here.
	dir = camera_matrix.block<3, 3>(0, 0) * dir;

	Eigen::Vector3f origin = camera_matrix.block<3, 3>(0, 0) * head_pos + camera_matrix.col(3);

	if (aperture_size != 0.0f) {
		Eigen::Vector3f lookat = origin + dir * focus_z;
		Eigen::Vector2f blur = aperture_size * square2disk_shirley(ld_random_val_2d(spp, uv.cwiseProduct(resolution.cast<float>()).cast<int>().dot(Eigen::Vector2i{19349663, 96925573})) * 2.0f - Eigen::Vector2f::Ones());
		origin += camera_matrix.block<3, 2>(0, 0) * blur;
		dir = (lookat - origin) / focus_z;
	}
	// screen_center.x() = 0.50, screen_center.y() = 0.50, near_distance = 0.00 

	origin += dir * near_distance;
	return {origin, dir};
}


inline NGP_HOST_DEVICE Ray pixel_to_ray(
	uint32_t spp,
	const Eigen::Vector2i& pixel,
	const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
	const Eigen::Vector2f& screen_center,
	const Eigen::Vector3f& parallax_shift = Eigen::Vector3f::Zero(),
	bool snap_to_pixel_centers = false,
	float near_distance = 0.0f,
	float focus_z = 1.0f,
	float aperture_size = 0.0f,
	const Foveation& foveation = {},
	Buffer2DView<const uint8_t> hidden_area_mask = {},
	const Lens& lens = {},
	Buffer2DView<const Eigen::Vector2f> distortion = {}
) {
	return uv_to_ray(
		spp,
		(pixel.cast<float>() + ld_random_pixel_offset(snap_to_pixel_centers ? 0 : spp)).cwiseQuotient(resolution.cast<float>()),
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		near_distance,
		focus_z,
		aperture_size,
		foveation,
		hidden_area_mask,
		lens,
		distortion
	);
}

inline NGP_HOST_DEVICE Eigen::Vector2f pos_to_uv(
	const Eigen::Vector3f& pos,
	const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
	const Eigen::Vector2f& screen_center,
	const Eigen::Vector3f& parallax_shift,
	const Foveation& foveation = {},
	const Lens& lens = {}
) {
	// Express ray in terms of camera frame
	Eigen::Vector3f head_pos = {parallax_shift.x(), parallax_shift.y(), 0.f};
	Eigen::Vector3f origin = camera_matrix.block<3, 3>(0, 0) * head_pos + camera_matrix.col(3);

	Eigen::Vector3f dir = pos - origin;
	dir = camera_matrix.block<3, 3>(0, 0).inverse() * dir;
	dir /= dir.z();
	dir += head_pos * parallax_shift.z();

	float du = 0.0f, dv = 0.0f;
	if (lens.mode == ELensMode::OpenCV) {
		opencv_lens_distortion_delta(lens.params, dir.x(), dir.y(), &du, &dv);
	} else if (lens.mode == ELensMode::OpenCVFisheye) {
		opencv_fisheye_lens_distortion_delta(lens.params, dir.x(), dir.y(), &du, &dv);
	} else {
		// No other type of distortion is permitted.
		assert(lens.mode == ELensMode::Perspective);
	}

	dir.x() += du;
	dir.y() += dv;

	Eigen::Vector2f uv = Eigen::Vector2f{dir.x(), dir.y()}.cwiseProduct(focal_length).cwiseQuotient(resolution.cast<float>()) + screen_center;
	return foveation.unwarp(uv);
}

inline NGP_HOST_DEVICE Eigen::Vector2f pos_to_pixel(
	const Eigen::Vector3f& pos,
	const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
	const Eigen::Vector2f& screen_center,
	const Eigen::Vector3f& parallax_shift,
	const Foveation& foveation = {},
	const Lens& lens = {}
) {
	return pos_to_uv(pos, resolution, focal_length, camera_matrix, screen_center, parallax_shift, foveation, lens).cwiseProduct(resolution.cast<float>());
}

inline NGP_HOST_DEVICE Eigen::Vector2f motion_vector(
	const uint32_t sample_index,
	const Eigen::Vector2i& pixel,
	const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera,
	const Eigen::Matrix<float, 3, 4>& prev_camera,
	const Eigen::Vector2f& screen_center,
	const Eigen::Vector3f& parallax_shift,
	const bool snap_to_pixel_centers,
	const float depth,
	const Foveation& foveation = {},
	const Foveation& prev_foveation = {},
	const Lens& lens = {}
) {
	Eigen::Vector2f pxf = pixel.cast<float>() + ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
	Ray ray = uv_to_ray(
		sample_index,
		pxf.cwiseQuotient(resolution.cast<float>()),
		resolution,
		focal_length,
		camera,
		screen_center,
		parallax_shift,
		0.0f,
		1.0f,
		0.0f,
		foveation,
		{}, // No hidden area mask
		lens
	);

	Eigen::Vector2f prev_pxf = pos_to_pixel(
		ray(depth),
		resolution,
		focal_length,
		prev_camera,
		screen_center,
		parallax_shift,
		prev_foveation,
		lens
	);

	return prev_pxf - pxf;
}

// Maps view-space depth (physical units) in the range [znear, zfar] hyperbolically to
// the interval [1, 0]. This is the reverse-z-component of "normalized device coordinates",
// which are commonly used in rasterization, where linear interpolation in screen space
// has to be equivalent to linear interpolation in real space (which, in turn, is
// guaranteed by the hyperbolic mapping of depth). This format is commonly found in
// z-buffers, and hence expected by downstream image processing functions, such as DLSS
// and VR reprojection.
inline NGP_HOST_DEVICE float to_ndc_depth(float z, float n, float f) {
	// View depth outside of the view frustum leads to output outside of [0, 1]
	z = tcnn::clamp(z, n, f);

	float scale = n / (n - f);
	float bias = -f * scale;
	return tcnn::clamp((z * scale + bias) / z, 0.0f, 1.0f);
}

inline NGP_HOST_DEVICE float fov_to_focal_length(int resolution, float degrees) {
	return 0.5f * (float)resolution / tanf(0.5f * degrees*(float)PI()/180);
}

inline NGP_HOST_DEVICE Eigen::Vector2f fov_to_focal_length(const Eigen::Vector2i& resolution, const Eigen::Vector2f& degrees) {
	return 0.5f * resolution.cast<float>().cwiseQuotient((0.5f * degrees * (float)PI()/180).array().tan().matrix());
}

inline NGP_HOST_DEVICE float focal_length_to_fov(int resolution, float focal_length) {
	return 2.f * 180.f / PI() * atanf(float(resolution)/(focal_length*2.f));
}

inline NGP_HOST_DEVICE Eigen::Vector2f focal_length_to_fov(const Eigen::Vector2i& resolution, const Eigen::Vector2f& focal_length) {
	return 2.f * 180.f / PI() * resolution.cast<float>().cwiseQuotient(focal_length*2).array().atan().matrix();
}

inline NGP_HOST_DEVICE float4 to_float4(const Eigen::Array4f& x) {
	return {x.x(), x.y(), x.z(), x.w()};
}

inline NGP_HOST_DEVICE float4 to_float4(const Eigen::Vector4f& x) {
	return {x.x(), x.y(), x.z(), x.w()};
}

inline NGP_HOST_DEVICE float3 to_float3(const Eigen::Array3f& x) {
	return {x.x(), x.y(), x.z()};
}

inline NGP_HOST_DEVICE float3 to_float3(const Eigen::Vector3f& x) {
	return {x.x(), x.y(), x.z()};
}

inline NGP_HOST_DEVICE float2 to_float2(const Eigen::Array2f& x) {
	return {x.x(), x.y()};
}

inline NGP_HOST_DEVICE float2 to_float2(const Eigen::Vector2f& x) {
	return {x.x(), x.y()};
}

inline NGP_HOST_DEVICE Eigen::Array4f to_array4(const float4& x) {
	return {x.x, x.y, x.z, x.w};
}

inline NGP_HOST_DEVICE Eigen::Vector4f to_vec4(const float4& x) {
	return {x.x, x.y, x.z, x.w};
}

inline NGP_HOST_DEVICE Eigen::Array3f to_array3(const float3& x) {
	return {x.x, x.y, x.z};
}

inline NGP_HOST_DEVICE Eigen::Vector3f to_vec3(const float3& x) {
	return {x.x, x.y, x.z};
}

inline NGP_HOST_DEVICE Eigen::Array2f to_array2(const float2& x) {
	return {x.x, x.y};
}

inline NGP_HOST_DEVICE Eigen::Vector2f to_vec2(const float2& x) {
	return {x.x, x.y};
}

inline NGP_HOST_DEVICE Eigen::Vector3f faceforward(const Eigen::Vector3f& n, const Eigen::Vector3f& i, const Eigen::Vector3f& nref) {
	return n * copysignf(1.0f, i.dot(nref));
}

inline NGP_HOST_DEVICE void apply_quilting(uint32_t* x, uint32_t* y, const Eigen::Vector2i& resolution, Eigen::Vector3f& parallax_shift, const Eigen::Vector2i& quilting_dims) {
	float resx = float(resolution.x()) / quilting_dims.x();
	float resy = float(resolution.y()) / quilting_dims.y();
	int panelx = (int)floorf(*x/resx);
	int panely = (int)floorf(*y/resy);
	*x = (*x - panelx * resx);
	*y = (*y - panely * resy);
	int idx = panelx + quilting_dims.x() * panely;

	if (quilting_dims == Eigen::Vector2i{2, 1}) {
		// Likely VR: parallax_shift.x() is the IPD in this case. The following code centers the camera matrix between both eyes.
		// idx == 0 -> left eye -> -1/2 x
		parallax_shift.x() = (idx == 0) ? (-0.5f * parallax_shift.x()) : (0.5f * parallax_shift.x());
	} else {
		// Likely HoloPlay lenticular display: in this case, `parallax_shift.z()` is the inverse height of the head above the display.
		// The following code computes the x-offset of views as a function of this.
		const float max_parallax_angle = 17.5f; // suggested value in https://docs.lookingglassfactory.com/keyconcepts/camera
		float parallax_angle = max_parallax_angle * PI() / 180.f * ((idx+0.5f)*2.f / float(quilting_dims.y() * quilting_dims.x()) - 1.f);
		parallax_shift.x() = atanf(parallax_angle) / parallax_shift.z();
	}
}

template <typename T>
__global__ void from_rgba32(const uint64_t num_pixels, const uint8_t* __restrict__ pixels, T* __restrict__ out, bool white_2_transparent = false, bool black_2_transparent = false, uint32_t mask_color = 0) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_pixels) return;

	uint8_t rgba[4];
	*((uint32_t*)&rgba[0]) = *((uint32_t*)&pixels[i*4]);

	float alpha = rgba[3] * (1.0f/255.0f);
	// NSVF dataset has 'white = transparent' madness
	if (white_2_transparent && rgba[0]==255 && rgba[1]==255 && rgba[2]==255) {
		alpha = 0.f;
	}
	if (black_2_transparent && rgba[0]==0 && rgba[1]==0 && rgba[2]==0) {
		alpha = 0.f;
	}

	tcnn::vector_t<T, 4> rgba_out;
	rgba_out[0] = (T)(srgb_to_linear(rgba[0] * (1.0f/255.0f)) * alpha);
	rgba_out[1] = (T)(srgb_to_linear(rgba[1] * (1.0f/255.0f)) * alpha);
	rgba_out[2] = (T)(srgb_to_linear(rgba[2] * (1.0f/255.0f)) * alpha);
	rgba_out[3] = (T)alpha;

	if (mask_color != 0 && mask_color == *((uint32_t*)&rgba[0])) {
		rgba_out[0] = rgba_out[1] = rgba_out[2] = rgba_out[3] = (T)-1.0f;
	}

	*((tcnn::vector_t<T, 4>*)&out[i*4]) = rgba_out;
}


// Foley & van Dam p593 / http://en.wikipedia.org/wiki/HSL_and_HSV
inline NGP_HOST_DEVICE Eigen::Array3f hsv_to_rgb(const Eigen::Array3f& hsv) {
	float h = hsv.x(), s = hsv.y(), v = hsv.z();
	if (s == 0.0f) {
		return Eigen::Array3f::Constant(v);
	}

	h = fmodf(h, 1.0f) * 6.0f;
	int i = (int)h;
	float f = h - (float)i;
	float p = v * (1.0f - s);
	float q = v * (1.0f - s * f);
	float t = v * (1.0f - s * (1.0f - f));

	switch (i) {
		case 0: return {v, t, p};
		case 1: return {q, v, p};
		case 2: return {p, v, t};
		case 3: return {p, q, v};
		case 4: return {t, p, v};
		case 5: default: return {v, p, q};
	}
}

inline NGP_HOST_DEVICE Eigen::Array3f to_rgb(const Eigen::Vector2f& dir) {
	return hsv_to_rgb({atan2f(dir.y(), dir.x()) / (2.0f * PI()) + 0.5f, 1.0f, dir.norm()});
}

enum class EImageDataType {
	None,
	Byte,
	Half,
	Float,
};

enum class EDepthDataType {
	UShort,
	Float,
};

inline NGP_HOST_DEVICE Eigen::Vector2i image_pos(const Eigen::Vector2f& pos, const Eigen::Vector2i& resolution) {
	return pos.cwiseProduct(resolution.cast<float>()).cast<int>().cwiseMin(resolution - Eigen::Vector2i::Constant(1)).cwiseMax(0);
}

inline NGP_HOST_DEVICE uint64_t pixel_idx(const Eigen::Vector2i& pos, const Eigen::Vector2i& resolution, uint32_t img) {
	return pos.x() + pos.y() * resolution.x() + img * (uint64_t)resolution.x() * resolution.y();
}

inline NGP_HOST_DEVICE uint64_t pixel_idx(const Eigen::Vector2f& xy, const Eigen::Vector2i& resolution, uint32_t img) {
	return pixel_idx(image_pos(xy, resolution), resolution, img);
}

// inline NGP_HOST_DEVICE Array3f composit_and_lerp(Vector2f pos, const Vector2i& resolution, uint32_t img, const __half* training_images, const Array3f& background_color, const Array3f& exposure_scale = Array3f::Ones()) {
// 	pos = (pos.cwiseProduct(resolution.cast<float>()) - Vector2f::Constant(0.5f)).cwiseMax(0.0f).cwiseMin(resolution.cast<float>() - Vector2f::Constant(1.0f + 1e-4f));

// 	const Vector2i pos_int = pos.cast<int>();
// 	const Vector2f weight = pos - pos_int.cast<float>();

// 	const Vector2i idx = pos_int.cwiseMin(resolution - Vector2i::Constant(2)).cwiseMax(0);

// 	auto read_val = [&](const Vector2i& p) {
// 		__half val[4];
// 		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
// 		return Array3f{val[0], val[1], val[2]} * exposure_scale + background_color * (1.0f - (float)val[3]);
// 	};

// 	return (
// 		(1 - weight.x()) * (1 - weight.y()) * read_val({idx.x(), idx.y()}) +
// 		(weight.x()) * (1 - weight.y()) * read_val({idx.x()+1, idx.y()}) +
// 		(1 - weight.x()) * (weight.y()) * read_val({idx.x(), idx.y()+1}) +
// 		(weight.x()) * (weight.y()) * read_val({idx.x()+1, idx.y()+1})
// 	);
// }

// inline NGP_HOST_DEVICE Array3f composit(Vector2f pos, const Vector2i& resolution, uint32_t img, const __half* training_images, const Array3f& background_color, const Array3f& exposure_scale = Array3f::Ones()) {
// 	auto read_val = [&](const Vector2i& p) {
// 		__half val[4];
// 		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
// 		return Array3f{val[0], val[1], val[2]} * exposure_scale + background_color * (1.0f - (float)val[3]);
// 	};

// 	return read_val(image_pos(pos, resolution));
// }

inline NGP_HOST_DEVICE Eigen::Array4f read_rgba(Eigen::Vector2i px, const Eigen::Vector2i& resolution, const void* pixels, EImageDataType image_data_type, uint32_t img = 0) {
	switch (image_data_type) {
		default:
			// This should never happen. Bright red to indicate this.
			return Eigen::Array4f{5.0f, 0.0f, 0.0f, 1.0f};
		case EImageDataType::Byte: {
			uint8_t val[4];
			*(uint32_t*)&val[0] = ((uint32_t*)pixels)[pixel_idx(px, resolution, img)];
			if (*(uint32_t*)&val[0] == 0x00FF00FF) {
				return Eigen::Array4f::Constant(-1.0f);
			}

			float alpha = (float)val[3] * (1.0f/255.0f);
			return Eigen::Array4f{
				srgb_to_linear((float)val[0] * (1.0f/255.0f)) * alpha,
				srgb_to_linear((float)val[1] * (1.0f/255.0f)) * alpha,
				srgb_to_linear((float)val[2] * (1.0f/255.0f)) * alpha,
				alpha,
			};
		}
		case EImageDataType::Half: {
			__half val[4];
			*(uint64_t*)&val[0] = ((uint64_t*)pixels)[pixel_idx(px, resolution, img)];
			return Eigen::Array4f{val[0], val[1], val[2], val[3]};
		}
		case EImageDataType::Float:
			return ((Eigen::Array4f*)pixels)[pixel_idx(px, resolution, img)];
	}
}

inline NGP_HOST_DEVICE Eigen::Array4f read_rgba(Eigen::Vector2f pos, const Eigen::Vector2i& resolution, const void* pixels, EImageDataType image_data_type, uint32_t img = 0) {
	return read_rgba(image_pos(pos, resolution), resolution, pixels, image_data_type, img);
}

inline NGP_HOST_DEVICE float read_depth(Eigen::Vector2f pos, const Eigen::Vector2i& resolution, const float* depth, uint32_t img = 0) {
	auto read_val = [&](const Eigen::Vector2i& p) {
		return depth[pixel_idx(p, resolution, img)];
	};

	return read_val(image_pos(pos, resolution));
}

Eigen::Matrix<float, 3, 4> log_space_lerp(const Eigen::Matrix<float, 3, 4>& begin, const Eigen::Matrix<float, 3, 4>& end, float t);

inline void depth_rgb(float depth, uint8_t *R, uint8_t *G, uint8_t *B)
{
	uint32_t rgb = (uint32_t)(depth * 512.0f);
	*R = (rgb & 0xff0000) >> 16;
	*G = (rgb & 0x00ff00) >> 8;
	*B = (rgb & 0xff);
}

inline void rgb_depth(uint8_t R, uint8_t G, uint8_t B, float *depth)
{
	uint32_t rgb = ((uint32_t)R << 16) | ((uint32_t)G << 8) | ((uint32_t)B);
	*depth = (float)rgb/512.0f;
}

tcnn::GPUMemory<float> load_exr_gpu(const fs::path& path, int* width, int* height);
tcnn::GPUMemory<float> load_stbi_gpu(const fs::path& path, int* width, int* height);
void save_stbi_gpu(const std::string& filename, int width, int height, Eigen::Array4f *gpu_rgba);
void save_depth_gpu(const std::string& filename, int width, int height, float *gpu_depth);

NGP_NAMESPACE_END
