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


int main_func(const std::vector<std::string>& arguments) {
	ArgumentParser parser{
		"Instant Neural Graphics Primitives Version " NGP_VERSION,
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
		{"no_gui"},
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

#ifdef NGP_GUI
	Flag vr_flag{
		parser,
		"VR",
		"Enable VR",
		{"vr"}
	};
#endif

	ValueFlag<string> load_config_flag{
		parser,
		"CONFIG_FILE",
		"Load network config from *.json file.",
		{"load_config"},
	};

	ValueFlag<string> load_model_flag{
		parser,
		"MODEL_FILE",
		"Load model from *.msgpack file.",
		{"load_model"},
	};

	ValueFlag<string> load_data_flag{
		parser,
		"DATASET_NAME",
		"Load training data from dataset (Folder for NeRF, *.obj/*.stl for SDF, *.nvdb for volume, others for image ).",
		{"load_data"},
	};

	ValueFlag<string> output_dir_flag{
		parser,
		"output",
		"Set output directory.",
		{"output"},
	};


	Flag save_model_flag{
		parser,
		"SAVE_MODEL",
		"Save model to output/model.msgpack.",
		{"save_model"},
	};

	Flag save_mesh_flag{
		parser,
		"SAVE_MESH",
		"Save mesh to output/mesh.obj for NeRF or SDF.",
		{"save_mesh"},
	};

	Flag save_images_flag{
		parser,
		"SAVE_IMAGES",
		"Render images/depth to output for NeRF.",
		{"save_images"},
	};

	ValueFlag<float> save_points_flag{
		parser,
		"S%",
		"Save point cloud (sample S%) to output/3d_pc.ply for NeRF.",
		{"save_points"},
	};

	Flag no_train_flag{
		parser,
		"NO_TRAIN",
		"Disable training.",
		{"no_train"},
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

	if (load_config_flag) {
		testbed.reload_network_from_file(get(load_config_flag));
	}

	if (load_model_flag) {
	    fs::path filename = get(load_model_flag);
	    load_model(testbed, filename);
	}

	testbed.m_train = !no_train_flag;

#ifdef NGP_GUI
	bool gui = !no_gui_flag;
#else
	bool gui = false;
#endif
	if (! testbed.m_train) // set no-gui without training
		gui = false;

	if (gui) {
		testbed.init_window(width_flag ? get(width_flag) : atoi(DEF_GUI_WIDTH),
			height_flag ? get(height_flag) : atoi(DEF_GUI_HEIGHT));
	}

#ifdef NGP_GUI
	if (vr_flag) {
		testbed.init_vr();
	}
#endif

	// Render/training loop
	float curr_psnr = 0.0f;
	std::time_t start_time = std::time(nullptr);
	float max_psnr = (max_psnr_flag)? get(max_psnr_flag) : atof(DEF_MAX_PSNR);
	uint32_t max_time = (max_time_flag)? get(max_time_flag) : atoi(DEF_MAX_TIME);
	uint32_t max_epoch = (max_epoch_flag)? get(max_epoch_flag) : atoi(DEF_MAX_EPOCH);

	testbed.m_training_step = 0;
	while (testbed.m_train && testbed.frame()) {
		if (testbed.m_training_step % 100 != 0)
			continue;

		curr_psnr = psnr(testbed.m_loss_scalar.val());
		tlog::info() << "iteration=" << testbed.m_training_step 
				<< " loss=" << testbed.m_loss_scalar.val()
				<< " psnr=" << curr_psnr
				<< " memory=" << testbed.gpu_memory_used();

		// Training stop ?
		if (testbed.m_training_step >= max_epoch || curr_psnr >= max_psnr 
			|| (std::time(nullptr) - start_time) >= max_time) {
			break;
		}
	}

	fs::path output_dir = "output";
	if (output_dir_flag || save_model_flag || save_images_flag || save_mesh_flag || save_points_flag) {
		output_dir = output_dir_flag ? get(output_dir_flag) : "output";
		if (! output_dir.is_directory()) {
			fs::create_directory(output_dir);
		}
	}

	if (save_model_flag) {
	    fs::path filename = output_dir/"model.msgpack";
		testbed.save_snapshot(filename, false /*include_optimizer_state*/);
	}

	if (save_mesh_flag) {
		if (testbed.m_testbed_mode != ETestbedMode::Nerf && 
			testbed.m_testbed_mode != ETestbedMode::Sdf) {
			tlog::warning() << "Save mesh only for NeRF or SDF.";
		} else {
		    fs::path filename = output_dir/"mesh.obj";
			float thresh = (testbed.m_testbed_mode == ETestbedMode::Nerf)? 2.5f : 0.0f;
			testbed.compute_and_save_marching_cubes_mesh(filename.str().c_str(),
				Eigen::Vector3i{256, 256, 256}, {} /*BoundingBox*/, thresh, false /*uv_flag*/);
		}
	}

	if (save_images_flag) {
		if (testbed.m_testbed_mode != ETestbedMode::Nerf) {
			tlog::warning() << "Render images/depth only for NeRF.";
		} else {
			testbed.save_nerf_images(output_dir);
		}
	}

	if (save_points_flag) {
		if (testbed.m_testbed_mode != ETestbedMode::Nerf) {
			tlog::warning() << "Save point cloud only for NeRF.";
		} else {
			float ratio = get(save_points_flag);
		    fs::path filename = output_dir/"3d_pc.ply";
			if (ratio < 0.0f)
				ratio = 100.0f;
			if (ratio > 100.0f)
				ratio = 100.0f;

			testbed.save_nerf_points(ratio, filename.str().c_str());
		}
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
