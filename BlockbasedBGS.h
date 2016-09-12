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
	int block_height;				//��߶�
	int block_width;				//����
	Size block;						//��Լ���
	long block_size;				//���С
	int threshold_num_pixel;		//���ظ���
	int frame_duration;				//��Ƶ���еĳ���
	double threshold_block;			//ǰ��֮֡��
	double threshold_pixel;			//����֮��
	int num_block_cols;				//��ĸ���
	int num_block_rows;				//��ĸ���
	Size block_img_dims;			//��Լ���
	long long frameNumber;			//֡��

	Mat blockBackground;			//�鱳��
	Mat last_blockImage;			//��һ֡��ͼ��
	Mat cur_blockImage;				//��ǰ֡��ͼ��
	Mat weight_blockBackground;		//�鱳��Ȩ��
	Mat tmp_blockBackground;		//��ʱ�鱳��
	Mat mframe_duration;			//��Ƶ���еĳ���

	Mat cur_RGBImage;				//��ǰRGBͼ��
	Mat cur_GrayImage;				//��ǰ�Ҷ�ͼ
	Mat Background;					//����
	Mat Foreground;					//ǰ��
	Mat tmp_Background;				//��ʱ����
	Mat tmp_Foreground;				//��ʱǰ��

	Mat weight_blockForeground;		//��ǰ��Ȩ��
	Mat end_blockForeground;		//��ʱǰ���յ�
	Mat flag_blockinitBackground;	//����ģ�ͱ�ʶ

	Mat img_edge;					//��Ե��Ϣ
	Mat Foreground_binary;			//��ֵ��ǰ��ͼ
	Mat Foreground_preprocess;		//��ȥ��Ե��ǰ��ͼ

	blockGMM* m_modes;					//��˹���ģ��
	int max_modes;					//����ģ����
	float m_bg_threshold;			//Ȩ�شﵽҪ����ܳ�Ϊ����
	float sigma_para;				//��sigma
	float learningrate;				//ѧϰ��
	float m_variance;				//�¸�ֵ����
	Mat num_pixel_modes;			//ÿ�����ص��ģ����

	Mat heatMap;					//threshold����ͼ

private:
	Mat blockIntegral(Mat src, Mat& des, int width);

public:
	BlockbasedBGS(void);
	~BlockbasedBGS(void);
	void InitPara(const Size block, const Size image, const int frame_duration, const int threshold_pixel);
	void InitModel(const Mat& input_img, const Mat& block_img, const long long frameNumber);			//��ʼ��������
	void ResetModel(void);																		//���ø�����
	void SetRGBInputImage(const Mat& input_img, const Mat& grey_img, const long long frameNumber);					//����ͼ��
	void SetBlockImage(const Mat& block_img);													//�����ͼ��
	void process(void);																			//�������
	void ResetLastBlockImage(const Mat& block_img);												//������һ֡ͼ��
	void GetBackground(Mat& img_background);													//
	void GetForeground(Mat& img_foreground);
	void ShowHeatMap();
	//�õ�ͼ���Ե
	void GetEdgeImg(Mat& gray_img);																
	//����ǰ��ͼƬ
	void saveForegroundImg(int _frameNumber);
	void preprocessForegroundImg(int frameNumber);		//Ԥ����ǰ�������ɷֿ�ͼ��
	float GussianVarience(long pos, const int& pixel, const float& Alpha, int& numModes);
};