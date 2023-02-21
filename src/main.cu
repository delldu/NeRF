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

float psnr(float x) {
	return -10.f*logf(x)/logf(10.f);
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

	ValueFlag<string> mode_flag{
		parser,
		"MODE",
		"Deprecated. Do not use.",
		{'m', "mode"},
	};

	ValueFlag<string> network_config_flag{
		parser,
		"CONFIG",
		"Path to the network config. Uses the scene's default if unspecified.",
		{'n', 'c', "network", "config"},
	};

	Flag no_gui_flag{
		parser,
		"NO_GUI",
		"Disables the GUI and instead reports training progress on the command line.",
		{"no-gui"},
	};

#ifdef NGP_GUI
	Flag vr_flag{
		parser,
		"VR",
		"Enables VR",
		{"vr"}
	};
#endif

	Flag no_train_flag{
		parser,
		"NO_TRAIN",
		"Disables training on startup.",
		{"no-train"},
	};

	ValueFlag<string> scene_flag{
		parser,
		"SCENE",
		"The scene to load. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.",
		{'s', "scene"},
	};

	ValueFlag<string> load_snapshot_flag{
		parser,
		"SNAPSHOT",
		"Load snapshot upon startup.",
		{"load_snapshot"},
	};

	ValueFlag<string> save_snapshot_flag{
		parser,
		"SNAPSHOT",
		"Save snapshot to file at end.",
		{"save_snapshot"},
	};

	ValueFlag<uint32_t> width_flag{
		parser,
		"WIDTH",
		"Resolution width of the GUI.",
		{"width"},
	};

	ValueFlag<uint32_t> height_flag{
		parser,
		"HEIGHT",
		"Resolution height of the GUI.",
		{"height"},
	};

	ValueFlag<int32_t> max_epoch_flag{
		parser,
		"MAX_EPOCH",
		"Training stop if epoch >= max_epoch.",
		{"max_epoch"},
	};

	ValueFlag<int32_t> max_time_flag{
		parser,
		"MAX_TIME",
		"Training stop if time >= max_time.",
		{"max_time"},
	};

	ValueFlag<float> max_psnr_flag{
		parser,
		"MAX_PSNR",
		"Training stop if PSNR >= max_psnr.",
		{"max_psnr"},
	};

	Flag version_flag{
		parser,
		"VERSION",
		"Display the version.",
		{'v', "version"},
	};

	PositionalList<string> files{
		parser,
		"files",
		"Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.",
	};

	// Parse command line arguments and react to parsing
	// errors using exceptions.
	try {
		if (arguments.empty()) {
			tlog::error() << "Number of arguments must be bigger than 0.";
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

	if (mode_flag) {
		tlog::warning() << "The '--mode' argument is no longer in use. It has no effect. The mode is automatically chosen based on the scene.";
	}

	Testbed testbed;

	for (auto file : get(files)) {
		testbed.load_file(file);
	}

	if (scene_flag) {
		testbed.load_training_data(get(scene_flag));
	}

	if (load_snapshot_flag) {
	    fs::path snapshot_path = get(load_snapshot_flag);
	    if (snapshot_path.exists() && equals_case_insensitive(snapshot_path.extension(), "msgpack")) {
			testbed.load_snapshot(snapshot_path);
	    }
	} else if (network_config_flag) {
		testbed.reload_network_from_file(get(network_config_flag));
	}

	testbed.m_train = !no_train_flag;

#ifdef NGP_GUI
	bool gui = !no_gui_flag;
#else
	bool gui = false;
#endif

	if (gui) {
		testbed.init_window(width_flag ? get(width_flag) : 1920, height_flag ? get(height_flag) : 1080);
	}

#ifdef NGP_GUI
	if (vr_flag) {
		testbed.init_vr();
	}
#endif

	// Render/training loop
	float current_psnr = 0;
	std::time_t start_time = std::time(nullptr);

	while (testbed.frame()) {
		current_psnr = psnr(testbed.m_loss_scalar.val());

		if (! gui) {
			tlog::info() << "iteration=" << testbed.m_training_step 
				<< " loss=" << testbed.m_loss_scalar.val()
				<< " psnr=" << current_psnr;
		}

		// Training stop condition
		if (max_epoch_flag && testbed.m_training_step >= get(max_epoch_flag)) {
			break;
		}
		if (max_psnr_flag && current_psnr >= get(max_psnr_flag)) {
			break;
		}
		if (max_time_flag && (std::time(nullptr) - start_time) >= get(max_time_flag)) {
			break;
		}
	}

	if (save_snapshot_flag) {
	    fs::path snapshot_path = get(save_snapshot_flag);
		if (equals_case_insensitive(snapshot_path.extension(), "msgpack")) {
			testbed.save_snapshot(snapshot_path, false /*optimize state*/);
		} else {
			tlog::warning() << "Snapshot file extension should be 'msgpack'";
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
