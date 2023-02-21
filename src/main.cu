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
		"WIDTH",
		"Resolution GUI width.",
		{"width"},
	};

	ValueFlag<uint32_t> height_flag{
		parser,
		"HEIGHT",
		"Resolution GUI height.\n",
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
		"CONFIG",
		"Network config file, using default if unspecified.",
		{"config"},
	};

	ValueFlag<string> load_model_flag{
		parser,
		"MODEL",
		"Load model from *.msgpack file.",
		{"load_model"},
	};

	ValueFlag<string> save_model_flag{
		parser,
		"MODEL",
		"Save model to *.msgpack file.\n",
		{"save_model"},
	};

	ValueFlag<string> load_dataset_flag{
		parser,
		"DATASET",
		"Load dataset. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.\n",
		{"dataset"},
	};

	Flag no_train_flag{
		parser,
		"NO_TRAIN",
		"Disable training.",
		{"no-train"},
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
		"Training stop if time >= max_time seconds.",
		{"max_time"},
	};

	ValueFlag<float> max_psnr_flag{
		parser,
		"MAX_PSNR",
		"Training stop if PSNR >= max_psnr.\n",
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

	if (load_dataset_flag) {
		testbed.load_training_data(get(load_dataset_flag));
	}

	if (load_model_flag) {
	    fs::path snapshot = get(load_model_flag);
	    if (snapshot.exists() && equals_case_insensitive(snapshot.extension(), "msgpack")) {
			testbed.load_snapshot(snapshot);
	    } else {
			tlog::warning() << "Model file should be '*.msgpack' and exists.";
	    }
	}
	if (load_config_flag) {
		testbed.reload_network_from_file(get(load_config_flag));
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

// #ifdef NGP_GUI
// 	if (vr_flag) {
// 		testbed.init_vr();
// 	}
// #endif

	// Render/training loop
	float current_psnr = 0;
	std::time_t start_time = std::time(nullptr);

	while (testbed.m_train && testbed.frame()) {
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

	if (save_model_flag) {
	    fs::path snapshot = get(save_model_flag);
		if (equals_case_insensitive(snapshot.extension(), "msgpack")) {
			testbed.save_snapshot(snapshot, false /*include_optimizer_state*/);
		} else {
			tlog::warning() << "Model file should be '*.msgpack'.";
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
