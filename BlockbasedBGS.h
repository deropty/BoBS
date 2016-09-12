#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "../IBGS.h"

struct Block{
	Point2i point;
	Mat data;
};


typedef struct GMMGaussian
{
	float variance;
	float pixel;
	float weight;
	float significants;		// this is equal to weight / standard deviation and is used to
						// determine which Gaussians should be part of the background model
}blockGMM;

class BlockbasedBGS
{
private:
	int block_height;				//块高度
	int block_width;				//块宽度
	Size block;						//简约表达
	long block_size;				//块大小
	int threshold_num_pixel;		//像素个数
	int frame_duration;				//视频序列的长度
	double threshold_block;			//前后帧之间
	double threshold_pixel;			//像素之间
	int num_block_cols;				//块的个数
	int num_block_rows;				//块的个数
	Size block_img_dims;			//简约表达
	long long frameNumber;			//帧号

	Mat blockBackground;			//块背景
	Mat last_blockImage;			//上一帧块图像
	Mat cur_blockImage;				//当前帧块图像
	Mat weight_blockBackground;		//块背景权重
	Mat tmp_blockBackground;		//临时块背景
	Mat mframe_duration;			//视频序列的长度

	Mat cur_RGBImage;				//当前RGB图像
	Mat cur_GrayImage;				//当前灰度图
	Mat Background;					//背景
	Mat Foreground;					//前景
	Mat tmp_Background;				//临时背景
	Mat tmp_Foreground;				//临时前景

	Mat weight_blockForeground;		//块前景权重
	Mat end_blockForeground;		//临时前景终点
	Mat flag_blockinitBackground;	//初建模型标识

	Mat img_edge;					//边缘信息
	Mat Foreground_binary;			//二值化前景图
	Mat Foreground_preprocess;		//减去边缘的前景图

	blockGMM* m_modes;					//高斯混合模型
	int max_modes;					//最大的模型数
	float m_bg_threshold;			//权重达到要求才能成为背景
	float sigma_para;				//几sigma
	float learningrate;				//学习率
	float m_variance;				//新赋值方差
	Mat num_pixel_modes;			//每个像素点的模型数

	Mat heatMap;					//threshold热力图

private:
	Mat blockIntegral(Mat src, Mat& des, int width);

public:
	BlockbasedBGS(void);
	~BlockbasedBGS(void);
	void InitPara(const Size block, const Size image, const int frame_duration, const int threshold_pixel);
	void InitModel(const Mat& input_img, const Mat& block_img, const long long frameNumber);			//初始化各参数
	void ResetModel(void);																		//重置各参数
	void SetRGBInputImage(const Mat& input_img, const Mat& grey_img, const long long frameNumber);					//输入图像
	void SetBlockImage(const Mat& block_img);													//输入块图像
	void process(void);																			//处理过程
	void ResetLastBlockImage(const Mat& block_img);												//重置上一帧图像
	void GetBackground(Mat& img_background);													//
	void GetForeground(Mat& img_foreground);
	void ShowHeatMap();
	//得到图像边缘
	void GetEdgeImg(Mat& gray_img);																
	//保存前景图片
	void saveForegroundImg(int _frameNumber);
	void preprocessForegroundImg(int frameNumber);		//预处理前景，生成分块图像
	float GussianVarience(long pos, const int& pixel, const float& Alpha, int& numModes);
};