/*
 * Copyright (C) Christian Briones, 2013
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#include "Spectrograph.h"
#include "Utility.h"

#include <iostream>
#include <string>

#include <fstream>
#include <iostream>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

#include <atomic>         // std::atomic
#include <thread>         // std::thread
#include <vector>         // std::vector

void createSpectrographs(const std::string& sParentDirectory, const std::string& sOutputDirectory)
{
	for (const auto& p : fs::directory_iterator(sParentDirectory))
	{
		std::string sWavFile = p.path().string();
		Spectrograph spectrograph(sWavFile, 256, 256);

		if (!spectrograph.file_is_valid()) {
			exit(1);
		}

		spectrograph.set_window(Utility::blackman_harris);
		spectrograph.compute(2048, 0.8);

		// compute output file name
		size_t firstIndex = sWavFile.find_last_of("/");
		size_t lastIndex = sWavFile.find_last_of(".");
		std::string wavFileNoExtension = sWavFile.substr(firstIndex, lastIndex);
		spectrograph.save_image(sOutputDirectory + wavFileNoExtension + ".png", false);
	}
}

void createSingleSpetrograph(const std::string& sWavFile, const std::string& sOutputDirectory)
{
	Spectrograph spectrograph(sWavFile, 256, 256);

	if (!spectrograph.file_is_valid()) {
		exit(1);
	}

	spectrograph.set_window(Utility::blackman_harris);
	spectrograph.compute(2048, 0.8);

	// compute output file name
	size_t firstIndex = sWavFile.find_last_of("/");
	size_t lastIndex = sWavFile.find_last_of(".");
	std::string wavFileNoExtension = sWavFile.substr(firstIndex, lastIndex);
	spectrograph.save_image(sOutputDirectory + wavFileNoExtension + ".png", false);
}
void createSpectrographsParallel(const std::string& sParentDirectory, const std::string& sOutputDirectory)
{
	std::vector<std::thread> threads;

	for (const auto& p : fs::directory_iterator(sParentDirectory))
	{
		std::string sWavFile = p.path().string();
		threads.push_back(std::thread(createSingleSpetrograph, sWavFile, sOutputDirectory));
	}

	for (auto& th : threads) th.join();
}

int main(int argc, const char *argv[])
{
	//// serial code
	//std::cout << "start training" << std::endl;
	//createSpectrographs("/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio-split/Training/", "/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/Visualization/Training/");
	//std::cout << "start validation" << std::endl;
	//createSpectrographs("/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio-split/Validation/", "/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/Visualization/Validation/");

	// =====================================================================================

	// parallel code
	std::cout << "start training" << std::endl;
	createSpectrographsParallel(
		"/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio-split/Training/",
		"/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/spectrograms/Training/");
	std::cout << "end training" << std::endl;

	std::cout << "start validation" << std::endl;
	createSpectrographsParallel(
		"/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio-split/Validation/",
		"/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/spectrograms/Validation/");
	std::cout << "end validation" << std::endl;

	// =====================================================================================

	//// very parallel code
	//std::thread training(createSpectrographsParallel,
	//	"/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio-split/Training/",
	//	"/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/spectrograms/Training/");
	//std::thread testing(createSpectrographsParallel,
	//	"/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio-split/Validation/",
	//	"/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/spectrograms/Validation/");
	//std::cout << "start training and testing" << std::endl;

	//// synchronize threads
	//training.join();
	//testing.join();

	//std::cout << "end training and testing";
}