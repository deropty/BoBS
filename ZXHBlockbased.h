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

	int block_height;				//��߶�
	int block_width;				//����
	int frame_duration;				//��Ƶ���еĳ���
	int threshold_pixel;			//���ز����ֵ
	Size block;						//���ά��
	Size img_dims;					//ͼ���ά��
	Size block_img_dims;			//��ͼ���ά��

public:
	ZXHBlockbased();
	~ZXHBlockbased();

	void process(const Mat &img_input, Mat &img_output, Mat &img_bgmodel);
	//���ɿ鱳��ģ��
	void preprocess(const Mat& img_input, Mat& block_frame, Size _block, Mat& grey_img);


private:
	//void finish(void);
	void saveConfig();
	void loadConfig();
};