// scrfd_debug.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include<string>
#include <opencv2/opencv.hpp>

#include "tinydir.h"
#include "libscrfd.h"
#include "libyoloface.h"


using namespace std;
cv::Mat limit_window_auto(cv::Mat img)
{
	cv::Mat img_vis;
#define WINDOW_IMSHOW_HEIGHT  1000.0
#define WINDOW_IMSHOW_WIDTH  1900.0
	int imgw = img.cols;
	int imgh = img.rows;
	float scale_ratio = 1.0;
	if (1.0 * imgw / imgh > WINDOW_IMSHOW_WIDTH / WINDOW_IMSHOW_HEIGHT) {
		scale_ratio = WINDOW_IMSHOW_WIDTH / imgw;
	}
	else {
		scale_ratio = WINDOW_IMSHOW_HEIGHT / imgh;
	}
	if (scale_ratio >= 1)return img.clone();
	cv::resize(img, img_vis, cv::Size(), scale_ratio, scale_ratio);
	return img_vis;
}



int main()
{
    std::cout << "Hello World!\n";


	tinydir_dir dir;
	int i;
	//string dirpath = "Z:\\Dataset\\hair\\image_unsplash";
	//string dirpath = "Z:\\workspace\\yoloface\\dataset\\crawler";
	//string dirpath = "Z:\\workspace\\yoloface\\dataset\\MAFA\\MAFA-20230903T134252Z-001\\MAFA\\train-images00\\images";
	//string dirpath = "Z:\\workspace\\yoloface\\dataset\\label_test";
	//string dirpath = "Z:\\workspace\\yoloface\\dataset\\crawler";
	//string dirpath = "Z:\\workspace\\yoloface\\dataset\\hair_zhedang";
	string dirpath = "Z:\\workspace\\yoloplate\\test1k";
	tinydir_open_sorted(&dir, dirpath.c_str());



	//SCRFD myScrfd;
	//myScrfd.load("500m_kps");
	//for (i = 0; i < dir.n_files; i++)
	//{
	//	tinydir_file file;
	//	tinydir_readfile_n(&dir, &file, i);
	//	//printf("%s", file.name);
	//	if (file.is_dir)
	//	{
	//		printf("/");
	//		continue;
	//	}
	//	string image_file = dirpath + "\\" + file.name;
	//	cout << "Processing " << image_file << endl;
	//	cv::Mat image = cv::imread(image_file);
	//	cv::resize(image, image, cv::Size(), 0.25, 0.25);
	//	cv::Mat imagergb = image.clone();
	//	cv::cvtColor(image, imagergb,cv::COLOR_BGR2RGB);
	//	std::vector<FaceObject>faceobjects;
	//	myScrfd.detect(imagergb, faceobjects);
	//	myScrfd.draw(image, faceobjects);

	//	cv::imshow("Image", limit_window_auto(image));
	//	if (27 == cv::waitKey(0)) return 0;

	//}
	//tinydir_close(&dir);


	const int target_sizes =640;
	//const float mean_vals[3] ={127.f, 127.f, 127.f};
	const float mean_vals[3] = { 0.f, 0.f, 0.f };
	const float norm_vals[3] ={1 / 255.f, 1 / 255.f, 1 / 255.f};

	YoloFace myYoloface;
	//myYoloface.load("best-sim", target_sizes, mean_vals, norm_vals);
	myYoloface.load("plate-exp20-sim", target_sizes, mean_vals, norm_vals);
    //myYoloface.load("yolov7-lite-e", target_sizes, mean_vals, norm_vals);
	for (i = 0; i < dir.n_files; i++)
	{
		tinydir_file file;
		tinydir_readfile_n(&dir, &file, i);
		//printf("%s", file.name);
		if (file.is_dir)
		{
			printf("/\n");
			continue;
		}
		//std::cout << file.extension <<"  "<< (string(file.extension) == "jpg") << std::endl;
		//system("pause");
		string fileext = string(file.extension);
		if (fileext != "jpg" && fileext != "png" && fileext != "jpeg")
		{
			continue;
		}
		//if(file.extension)
		//printf();
		


		string image_file = dirpath + "\\" + file.name;
		cout << "Processing " << image_file << endl;
		cv::Mat image = cv::imread(image_file);
		//cv::resize(image, image, cv::Size(), 0.25, 0.25);
		cv::Mat imagergb = image.clone();
		cv::cvtColor(image, imagergb, cv::COLOR_BGR2RGB);
		std::vector<Object>faceobjects;
		myYoloface.detect(imagergb, faceobjects);
		myYoloface.draw(image, faceobjects);

		cv::imshow("Image", limit_window_auto(image));
		if (27 == cv::waitKey(0)) break;

	}
	tinydir_close(&dir);



}


