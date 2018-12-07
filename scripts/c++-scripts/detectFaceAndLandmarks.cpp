#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"

#include <iostream>
#include <string>
#include <utility>
#include <experimental/filesystem>
#include <fstream>
#include <array>

namespace fs = std::experimental::filesystem;
using namespace std;
using namespace cv;
using namespace cv::face;

//const string trainingOutputPath = "../../data/faces/Training/"s;
//const string validationOutputPath = "../../data/faces/Validation/"s;
//array<string,2> outputPaths{ trainingOutputPath, validationOutputPath };
const string outputPath = "../../data/Testing/faces/"s;
array<string, 2> outputPaths{ outputPath };

const string face_cascade_name = "./haarcascade_frontalface_default.xml"s;
CascadeClassifier face_cascade;

Ptr<Facemark> facemark = FacemarkLBF::create();
const string landmark_name = "./lbfmodel.yaml"s;

vector<fs::path> videoFilenames(const string& path) {
	vector<fs::path> filenames;
	for (auto& dir_entry : fs::recursive_directory_iterator(path)) {
		if (dir_entry.path().extension() == ".mp4") {
			filenames.push_back(dir_entry.path());
		}
	}
	return filenames;
}

//pair<vector<fs::path>, vector<fs::path>> trainingAndTestingFiles() {
//	return { videoFilenames("../../data/Testing/Videos/") };
//}
vector<fs::path> trainingAndTestingFiles() {
	//return { videoFilenames("../../data/Training/Videos/"), videoFilenames("../../data/Validation/Videos/") };
	return videoFilenames("../../data/Testing/Videos/");
}

Mat preprocess(Mat&, const bool = true);
Mat sizeTo256(Mat&& img);

vector<Mat> readVideo(const string& path, const bool subject, bool& flag) {
	vector<Mat> frames;
	VideoCapture cap(path);
	Mat frame;
	bool success = true;
	int i = 0;
	while (success && i < 250000) {
		success = cap.read(frame);
		if (!frame.empty()) {
			frames.push_back(sizeTo256(preprocess(frame, subject)));
		}
		++i;
	}
	if (!success) { flag = false; }
	return frames;
}

vector<Rect> detectFace(const Mat& frame, CascadeClassifier& face_cascade) {
	vector<Rect> faces;
	face_cascade.detectMultiScale(frame, faces, 1.5, 0, 3, Size(30, 30));
	return faces;
}

Mat preprocess(Mat& frame, const bool subject) {
	if (subject) {
		frame = frame(Rect(frame.size().width / 2, 0, frame.size().width / 2, frame.size().height));
	}
	else {
		frame = frame(Rect(0, 0, frame.size().width / 2, frame.size().height));
	}
	/*Mat gray;
	vector<Rect> faces;
	if (frame.channels() > 1) {
		cvtColor(frame, gray, COLOR_BGR2GRAY);
	}
	else {
		gray = frame.clone();
	}
	equalizeHist(gray, gray);*/
	return frame;
}

Mat grayToColor(Mat&& gray) {
	cvtColor(gray, gray, COLOR_GRAY2BGR);
	return gray;
}

Mat sizeTo256(Mat&& img) {
	resize(img, img, Size(256, 256), 0, 0, INTER_LINEAR_EXACT);
	return img;
}

vector<vector<Point2f>> detectFacesAndLandmarks(vector<Mat>& frames, CascadeClassifier& faceClassifier, Ptr<Facemark>& facemark, long& num_errors) {
	vector<vector<Point2f>> landmarks;
	landmarks.reserve(2048);
	vector<vector<Point2f>> landmark;

	int i = 0;
	int failure_count = 0;

	for (auto& frame : frames) {
		auto bounds = detectFace(frame, faceClassifier);
		if (bounds.empty()) {
			if (i == failure_count) {
				++failure_count;
				++num_errors;
			}
			else {
				landmarks.push_back(landmarks.back());
			}
		}
		else {
			if (facemark->fit(frame, bounds, landmark)) {
				landmarks.push_back(landmark.at(0));
				while (failure_count > 0) {
					landmarks.insert(landmarks.begin(), landmarks.back());
					--failure_count;
				}
			}
			else {
				if (i == failure_count) {
					++failure_count;
					++num_errors;
				}
				else {
					landmarks.push_back(landmarks.back());
				}
			}
		}
		++i;
	}
	return landmarks;
}

int main(int argc, char** argv) {
	face_cascade.load(face_cascade_name);
	facemark->loadModel(landmark_name);
	//const auto&[train, test] = trainingAndTestingFiles();
	const auto& test = trainingAndTestingFiles();
	//array<vector<fs::path>, 2> paths{ train, test };
	array<vector<fs::path>, 1> paths{ test };
	ofstream error_file("../../data/Testing/faces/errors.txt", ios::app);
	for (auto i = 0; i < 2; ++i) {
		bool subject;
		if (i == 0) { subject = true; }
		else { subject = false; }
		string end_string = ".landmarks.txt";
		if (!subject) { end_string = ".landmarks_actor.txt"; }
		//for (auto j = 0; j < 2; ++j) {
		for (auto j = 0; j < 1; ++j) {
			double total_errors = 0.0;
			long iteration_number = 0;
			for (const auto& path : paths.at(j)) {
				cout << "Reading Video at path: " << path.string() << '\n';
				bool flag = true;
				int idx = 0;
				while (flag) {
					auto frames = readVideo(path.string(), subject, flag);
					iteration_number += frames.size();
					long error_count = 0;
					cout << "Detecting Landmarks for batch " << idx << "!\n";
					auto landmarks = detectFacesAndLandmarks(frames, face_cascade, facemark, error_count);
					total_errors += error_count;
					ofstream of(outputPaths.at(j) + path.filename().string() + end_string, ios::app);
					cout << "Writing batch " << idx << " at path: " << outputPaths.at(j) + path.filename().string() + end_string << '\n';
					for (const auto& landmark : landmarks) {
						for (const auto& point : landmark) {
							of << point.x << ',' << point.y << ' ';
						}
						of << '\n';
					}
					++idx;
				}
			}
			error_file << total_errors / static_cast<double>(iteration_number) << "%\n";
		}
	}
}
