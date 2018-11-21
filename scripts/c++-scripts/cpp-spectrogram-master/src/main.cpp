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

int main(int argc, const char *argv[])
{
	//if (argc < 2){
	//    std::cout << "You must specify an input file." << std::endl;
	//    return -1;
	//}
	//std::string fname(argv[1]);
	//std::string fname = "/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio-split/Training/Subject_6_Story_4_frame_5640.wav";

	//Spectrograph spectrograph(fname, 256, 256);

	//if (!spectrograph.file_is_valid()) {
	//	return -1;
	//}
	//spectrograph.set_window(Utility::blackman_harris);
	//spectrograph.compute(2048, 0.8);
	//spectrograph.save_image("spectrogram-small-3.png", false);
	//return 0;

	std::cout << "start training" << std::endl;
	createSpectrographs("/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio-split/Training/", "/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/Visualization/Training/");
	std::cout << "start validation" << std::endl;
	createSpectrographs("/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio-split/Validation/", "/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/Visualization/Validation/");
}
