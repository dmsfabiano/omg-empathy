#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"

#include <iostream>
#include <string>
#include <utility>
#include <experimental/filesystem>
#include <fstream>

namespace fs = std::experimental::filesystem;
using namespace std;
using namespace cv;
using namespace cv::face;

const string trainingOutputPath = "../../data/faces/Training/"s;
const string validationOutputPath = "../../data/faces/Validation/"s;

const string face_cascade_name = "./haarcascade_frontalface_default.xml"s;
CascadeClassifier face_cascade;

Ptr<Facemark> facemark = FacemarkLBF::create();
const string landmark_name = "./lbfmodel.yaml"s;

vector<fs::path> videoFilenames(const string& path) {
	vector<fs::path> filenames;
	for(auto& dir_entry : fs::recursive_directory_iterator(path)) {
		if(dir_entry.path().extension() == ".mp4") {
			filenames.push_back(dir_entry.path());
		}
	}
	return filenames;
}

pair<vector<fs::path>, vector<fs::path>> trainingAndTestingFiles() {
	return {videoFilenames("../../data/Training/Videos/"), videoFilenames("../../data/Validation/Videos/")};
}

Mat preprocess(Mat&, const bool = true);

vector<Mat> readVideo(const string& path) {
	vector<Mat> frames;
	VideoCapture cap(path);
	Mat frame;
	bool success = true;
	while(success) {
		success = cap.read(frame);
		if(!frame.empty()) {
			frames.push_back(preprocess(frame));
		}
	}
	return frames;
}

vector<Rect> detectFace(const Mat& frame) {
	vector<Rect> faces;
	face_cascade.detectMultiScale( frame, faces, 1.4, 0, 6, Size(96, 96) );
	return faces;
}

Mat preprocess(Mat& frame, const bool subject) {
	if(subject) {
		frame = frame(Rect(frame.size().width/2, 0, frame.size().width/2, frame.size().height));
	} else {
		frame = frame(Rect(0, 0, frame.size().width/2, frame.size().height));
	}
	/*Mat gray;
	vector<Rect> faces;
	if(frame.channels()>1){
	    cvtColor(frame,gray,COLOR_BGR2GRAY);
	}
	else{
	    gray = frame.clone();
	}
	equalizeHist( gray, gray );
	return gray;*/
	return frame;
}

Mat grayToColor(Mat&& gray) {
	cvtColor(gray, gray, COLOR_GRAY2BGR);
	return gray;
}

Mat sizeTo256(Mat&& img) {
	resize(img,img,Size(256,256),0,0,INTER_LINEAR_EXACT);
	return img;
}

pair<vector<Mat>, vector<vector<Point2f>>> detectFacesAndLandmarks(vector<Mat>& frames) {
	vector<Mat> faces;
	vector<vector<Point2f>> landmarks;

	vector<vector<Point2f>> landmark;

	face_cascade.load(face_cascade_name);
	facemark->loadModel(landmark_name);

	double sum = 0.0;
	double errors = 0.0;
	double lm_error = 0.0;
	for(auto& frame : frames) {
		++sum;
		auto bounds = detectFace(frame);
		if(bounds.empty()) {
			++errors;
			if(!faces.empty()) {
				faces.push_back(faces.back());
				landmarks.push_back(landmarks.back());
			}
		} else {
			if(facemark->fit(frame,bounds,landmark)) {
				faces.push_back(sizeTo256(Mat(frame,bounds.at(0))));
				landmarks.push_back(landmark.at(0));
			} else {
				++lm_error;
				faces.push_back(sizeTo256(Mat(frame,bounds.at(0))));
				landmarks.push_back(landmarks.back());
			}
		}
	}
	ofstream out_file("accuracy.txt", ios::app);
	out_file << errors/sum * 100.0 << "%\t" << lm_error/sum * 100.0 << "%\n";
	return {faces,landmarks};
}

int main() {
	const auto& [train, test] = trainingAndTestingFiles();
	for(const auto& path : train) {
		auto frames = readVideo(path.string());
		auto [faces, landmarks] = detectFacesAndLandmarks(frames);
		unsigned int frame_number = 1;
		for(auto& face : faces) {
			imwrite(trainingOutputPath + path.filename().string() + "_frame" + to_string(frame_number) + ".png", face);
			++frame_number;
		}
		ofstream of(trainingOutputPath + path.filename().string() + ".landmarks.txt");
		for(const auto& landmark : landmarks) {
			for(const auto& point : landmark) {
				of << point.x << ',' << point.y << ' ';
			}
			of << '\n';
		}
	}
}
