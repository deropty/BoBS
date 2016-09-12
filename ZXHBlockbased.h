#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "../IBGS.h"
#include "BlockbasedBGS.h"

class ZXHBlockbased : public IBGS
{
private:
	bool firstTime;
	long long frameNumber;
	bool showOutput;
	bool loadDefaultParams;

	Mat img_foreground;
	Mat img_background;

	BlockbasedBGS* BGS;

	int block_height;				//块高度
	int block_width;				//块宽度
	int frame_duration;				//视频序列的长度
	int threshold_pixel;			//像素差的阈值
	Size block;						//块的维度
	Size img_dims;					//图像的维度
	Size block_img_dims;			//块图像的维度

public:
	ZXHBlockbased();
	~ZXHBlockbased();

	void process(const Mat &img_input, Mat &img_output, Mat &img_bgmodel);
	//生成块背景模型
	void preprocess(const Mat& img_input, Mat& block_frame, Size _block, Mat& grey_img);


private:
	//void finish(void);
	void saveConfig();
	void loadConfig();
};