/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   main.cu
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/testbed.h>

#include <tiny-cuda-nn/common.h>

#include <args/args.hxx>

#include <filesystem/path.h>

using namespace args;
using namespace ngp;
using namespace std;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

#define DEF_GUI_WIDTH "1920"
#define DEF_GUI_HEIGHT "1080"
#define DEF_MAX_TIME "36000" // 10 hours
#define DEF_MAX_PSNR "50.00"
#define DEF_MAX_EPOCH "1000000"

#define file_like(filename, likes) equals_case_insensitive(filename.extension(), likes)


inline float psnr(float x)
{
	return -10.f*logf(x)/logf(10.f);
}

void load_model(Testbed &testbed, fs::path &filename)
{
    if (! filename.exists()) {
		tlog::warning() << "Model file '" << filename.str() << "' does not exist";
		return;
    }

    if (! file_like(filename, "msgpack")) {
		tlog::warning() << "Model should be '*.msgpack' file";
		return;
    }

	testbed.load_snapshot(filename);
}

void save_model(Testbed &testbed, fs::path &filename)
{
	if (! file_like(filename, "msgpack")) {
		tlog::warning() << "Model should be '*.msgpack' file.";
		return;
	}
	// else
	testbed.save_snapshot(filename, false /*include_optimizer_state*/);
}


void save_mesh(Testbed &testbed, fs::path &filename)
{
    if (! file_like(filename, "obj")) {
		tlog::warning() << "Mesh should be *.obj file.";
		return;
    }

	if (testbed.m_testbed_mode != ETestbedMode::Nerf && 
		testbed.m_testbed_mode != ETestbedMode::Sdf) {
		tlog::warning() << "Save mesh only for NeRF or SDF.";
		return;
	}

	float thresh = (testbed.m_testbed_mode == ETestbedMode::Nerf)? 2.5f : 0.0f;
	// std::numeric_limits<float>::max()
	bool uv_flag = false; // file_like(filename, "obj"); /*generate_uvs_for_obj_file*/
	testbed.compute_and_save_marching_cubes_mesh(filename.str().c_str(),
		Eigen::Vector3i{256, 256, 256}, {} /*BoundingBox*/, thresh, uv_flag);
}

void save_point(Testbed &testbed, fs::path &filename)
{
    if (! file_like(filename, "png")) {
		tlog::warning() << "Point cloud should be *.ply file.";
		return;
    }

	if (testbed.m_testbed_mode != ETestbedMode::Nerf) {
		tlog::warning() << "Save point cloud only for NeRF.";
		return;
	}

	// testbed.get_nerf_rays_from_image(5);
	
	// testbed.save_nerf_point_cloud(filename.str().c_str());

	testbed.render_nerf_image(0, filename);
}


int main_func(const std::vector<std::string>& arguments) {
	ArgumentParser parser{
		"Instant Neural Graphics Primitives\n"
		"Version " NGP_VERSION,
		"",
	};

	HelpFlag help_flag{
		parser,
		"HELP",
		"Display this help.",
		{'h', "help"},
	};

	Flag version_flag{
		parser,
		"VERSION",
		"Display version.",
		{'v', "version"},
	};

	Flag no_gui_flag{
		parser,
		"NO_GUI",
		"Disable GUI.",
		{"no-gui"},
	};

	ValueFlag<uint32_t> width_flag{
		parser,
		DEF_GUI_WIDTH,
		"GUI width.",
		{"width"},
	};

	ValueFlag<uint32_t> height_flag{
		parser,
		DEF_GUI_HEIGHT,
		"GUI height.",
		{"height"},
	};

// #ifdef NGP_GUI
// 	Flag vr_flag{
// 		parser,
// 		"VR",
// 		"Enable VR",
// 		{"vr"}
// 	};
// #endif

	ValueFlag<string> load_config_flag{
		parser,
		"FILE_NAME",
		"Load net config from *.json file.",
		{"load_config"},
	};

	ValueFlag<string> load_model_flag{
		parser,
		"FILE_NAME",
		"Load model from *.msgpack file.",
		{"load_model"},
	};

	ValueFlag<string> save_model_flag{
		parser,
		"FILE_NAME",
		"Save model to *.msgpack file.",
		{"save_model"},
	};

	ValueFlag<string> save_mesh_flag{
		parser,
		"FILE_NAME",
		"Save mesh to *.obj file for NeRF or SDF.",
		{"save_mesh"},
	};

	ValueFlag<string> save_point_flag{
		parser,
		"FILE_NAME",
		"Save point cloud to *.ply file for NeRF.",
		{"save_point"},
	};

	ValueFlag<string> load_data_flag{
		parser,
		"DATASET",
		"Load training data from dataset (Folder for NeRF, *.obj/*.stl for SDF, *.nvdb for volume, others for image ).",
		{"load_data"},
	};

	Flag no_train_flag{
		parser,
		"NO_TRAIN",
		"Disable training.",
		{"no-train"},
	};

	ValueFlag<int32_t> max_epoch_flag{
		parser,
		DEF_MAX_EPOCH,
		"Training stop if epoch >= max_epoch.",
		{"max_epoch"},
	};

	ValueFlag<int32_t> max_time_flag{
		parser,
		DEF_MAX_TIME,
		"Training stop if time >= max_time seconds.",
		{"max_time"},
	};

	ValueFlag<float> max_psnr_flag{
		parser,
		DEF_MAX_PSNR,
		"Training stop if PSNR >= max_psnr.",
		{"max_psnr"},
	};

	PositionalList<string> files{
		parser,
		"files",
		"Files to be loaded. Can be a dataset, network config, snapshot, camera path, or a combination of those.",
	};

	// Parse command line arguments and react to parsing
	// errors using exceptions.
	try {
		if (arguments.empty()) {
			tlog::error() << "Argument number must be > 0.";
			return -3;
		}

		parser.Prog(arguments.front());
		parser.ParseArgs(begin(arguments) + 1, end(arguments));
	} catch (const Help&) {
		cout << parser;
		return 0;
	} catch (const ParseError& e) {
		cerr << e.what() << endl;
		cerr << parser;
		return -1;
	} catch (const ValidationError& e) {
		cerr << e.what() << endl;
		cerr << parser;
		return -2;
	}

	if (version_flag) {
		tlog::none() << "Instant Neural Graphics Primitives v" NGP_VERSION;
		return 0;
	}

	// Start ...
	Testbed testbed;

	for (auto file : get(files)) {
		testbed.load_file(file);
	}

	if (load_data_flag) {
		testbed.load_training_data(get(load_data_flag));
	}

	testbed.m_train = !no_train_flag;

	if (load_model_flag) {
	    fs::path filename = get(load_model_flag);
	    load_model(testbed, filename);
	}
	if (load_config_flag) {
		testbed.reload_network_from_file(get(load_config_flag));
	}


#ifdef NGP_GUI
	bool gui = !no_gui_flag;
#else
	bool gui = false;
#endif

	if (gui) {
		testbed.init_window(width_flag ? get(width_flag) : atoi(DEF_GUI_WIDTH),
			height_flag ? get(height_flag) : atoi(DEF_GUI_HEIGHT));
	}

// #ifdef NGP_GUI
// 	if (vr_flag) {
// 		testbed.init_vr();
// 	}
// #endif

	// Render/training loop
	float curr_psnr = 0.0f;
	std::time_t start_time = std::time(nullptr);
	float max_psnr = (max_psnr_flag)? get(max_psnr_flag) : atof(DEF_MAX_PSNR);
	uint32_t max_time = (max_time_flag)? get(max_time_flag) : atoi(DEF_MAX_TIME);
	uint32_t max_epoch = (max_epoch_flag)? get(max_epoch_flag) : atoi(DEF_MAX_EPOCH);

	testbed.redraw_gui_next_frame();
	while (testbed.frame()) {
		if (testbed.m_training_step % 100 != 0)
			continue;

		curr_psnr = psnr(testbed.m_loss_scalar.val());
		tlog::info() << "iteration=" << testbed.m_training_step 
				<< " loss=" << testbed.m_loss_scalar.val()
				<< " psnr=" << curr_psnr;

		// Training stop ?
		if (testbed.m_training_step >= max_epoch || curr_psnr >= max_psnr 
			|| (std::time(nullptr) - start_time) >= max_time) {
			break;
		}
	}

	if (save_model_flag) {
	    fs::path filename = get(save_model_flag);
	    save_model(testbed, filename);
	}

	if (save_mesh_flag) {
	    fs::path filename = get(save_mesh_flag);
	    save_mesh(testbed, filename);
	}

	if (save_point_flag) {
	    fs::path filename = get(save_point_flag);
	    save_point(testbed, filename);
	}

	return 0;
}

NGP_NAMESPACE_END

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
	SetConsoleOutputCP(CP_UTF8);
#else
int main(int argc, char* argv[]) {
#endif
	try {
		std::vector<std::string> arguments;
		for (int i = 0; i < argc; ++i) {
#ifdef _WIN32
			arguments.emplace_back(ngp::utf16_to_utf8(argv[i]));
#else
			arguments.emplace_back(argv[i]);
#endif
		}
		if (argc == 1) {
			arguments.emplace_back("--help");
		}

		return ngp::main_func(arguments);
	} catch (const exception& e) {
		tlog::error() << fmt::format("Uncaught exception: {}", e.what());
		return 1;
	}
}
