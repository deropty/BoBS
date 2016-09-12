#include "blockBasedBGS.h"
#include <string>
#include <ctime>

BlockbasedBGS::BlockbasedBGS()
{
	m_modes = NULL;
}

BlockbasedBGS::~BlockbasedBGS()
{
	delete[] m_modes;
}

void BlockbasedBGS::InitPara(const Size block_, const Size image_, const int duration, const int threshold_pixel_)
{
	block = block_;
	block_height = block.height;
	block_width = block.width;
	block_size = block_.area();
	frame_duration = duration;
	threshold_num_pixel = block_size / 2;
	threshold_pixel = threshold_pixel_;
	threshold_block = block_size * threshold_pixel_ / 2;		//抵消块带来的影响，采用减半的策略
	num_block_cols = image_.width / block_.width;
	num_block_rows = image_.height / block_.height;
	block_img_dims = Size(num_block_cols, num_block_rows);

	max_modes = 5;
	m_bg_threshold = 0.75f;		//nonsense
	sigma_para = 3;				//3sigma
	learningrate = 0.001f;
	m_variance = threshold_block / 3.0;
}

void BlockbasedBGS::InitModel(const Mat& input_img, const Mat& block_img, const long long frameNumber_)
{
	blockBackground = block_img.clone();
	tmp_blockBackground = block_img.clone();
	last_blockImage = block_img.clone();
	tmp_Background = input_img.clone();
	Background = input_img.clone();

	weight_blockBackground = Mat(block_img_dims, CV_32SC1, Scalar(0));
	weight_blockForeground = Mat(block_img_dims, CV_32SC1, Scalar(0));
	end_blockForeground = Mat(block_img_dims, CV_32SC1, Scalar(0));
	flag_blockinitBackground = Mat(block_img_dims, CV_32SC1, Scalar(0));
	mframe_duration = Mat(block_img_dims, CV_32SC1, Scalar(frame_duration));
	
	tmp_Foreground = Mat(input_img.size(), input_img.type(), Scalar(0, 0, 0));

	//heatMap = Mat(input_img.size(), CV_8UC1, Scalar(0));
	
	// num of modes for per pixel
	num_pixel_modes = Mat(block_img_dims, CV_32SC1, Scalar(0));
	// GMM for each pixel
	m_modes = new blockGMM[num_block_cols*num_block_rows*max_modes];
	for (unsigned int i = 0; i < num_block_cols*num_block_rows*max_modes; ++i)
	{
		m_modes[i].weight = 0;
		m_modes[i].variance = 0;
		m_modes[i].pixel = 0;
		m_modes[i].significants = 0;
	}

	frameNumber = frameNumber_;
}

void BlockbasedBGS::SetBlockImage(const Mat& block_img)
{
	cur_blockImage = block_img.clone();
}

void BlockbasedBGS::SetRGBInputImage(const Mat& input_img, const Mat& gray_img, const long long frameNumber_)
{
	Foreground = Mat(input_img.size(), CV_8UC3, Scalar(0, 0, 0));		//重置前景
	cur_RGBImage = input_img.clone();
	cur_GrayImage = gray_img.clone();
	frameNumber = frameNumber_;
}

void BlockbasedBGS::process()
{
	int cnt = 0;			//记录前景数
	std::srand(time(0));	//随机数种子
	for (int i = 0; i < num_block_rows; ++i){

		int* cur_bI = cur_blockImage.ptr<int>(i);
		int* last_bI = last_blockImage.ptr<int>(i);
		int* bB = blockBackground.ptr<int>(i);

		int* wt_bB = weight_blockBackground.ptr<int>(i);
		int* t_bB = tmp_blockBackground.ptr<int>(i);		//double类型会影响到下一个矩阵
		int* f = flag_blockinitBackground.ptr<int>(i);
		int* end_fg = end_blockForeground.ptr<int>(i);
		int* wt_bF = weight_blockForeground.ptr<int>(i);

		int* npm = num_pixel_modes.ptr<int>(i);				//每个像素独立更新模型数

		int* mfd = mframe_duration.ptr<int>(i);				//每个像素独立更新运动帧长

		for (int j = 0; j < num_block_cols; ++j){

			Rect rect(j*block_width, i*block_height, block_width, block_height);
			Mat roi_cur(cur_RGBImage, rect);
			Mat roi_Bg(Background, rect);
			Mat roi_t_Bg(tmp_Background, rect);

			Mat roi_Fg(Foreground, rect);
			Mat roi_t_Fg(tmp_Foreground, rect);
			Mat diff;		//背景roi与当前帧roi差值
			Mat diff_c1;	//单通道diff
			int num_pixel;

			//Mat roi_hm(heatMap, rect);

			int pos = i*num_block_cols + j;
			threshold_block = GussianVarience(pos, cur_bI[j], learningrate, npm[j]);

			//int heat = (threshold_block / sigma_para - 4.0) * 255.0 / (5.0 * m_variance - 4.0);
			//roi_hm = Scalar(heat);			//设置所有值为heat

			double tmp = abs(cur_bI[j] - last_bI[j]);
			if (tmp > threshold_block){
				wt_bB[j] = 0;
				t_bB[j] = cur_bI[j];
				/*RGB图像*/
				roi_cur.copyTo(roi_t_Bg);
			}else{	
				int wt = wt_bB[j];
				t_bB[j] = (t_bB[j] * wt + cur_bI[j]) / (wt + 1.0);
				wt_bB[j] ++;
				/*RGB图像*/
				Mat tmp;
				addWeighted(roi_t_Bg, wt / (wt + 1.0), roi_cur, 1.0 / (wt + 1.0), 0, tmp);	//使用tmp,防止可能新创建的对象
				tmp.copyTo(roi_t_Bg);
			}

			if (wt_bB[j] >= mfd[j]){		
				bB[j] = t_bB[j];
				if (f[j] == 0){
					mfd[j] *= 20;
					f[j] = 1;
				}
				/*RGB图像*/
				/*有待考虑，也许用所有的均值会更好，这里采用部分替换的方式*/
				roi_t_Bg.copyTo(roi_Bg);

				//if (frameNumber == 628){
				//	int a = 0;
				//	if (i == 72 && j == 41){
				//		char t[10];
				//		sprintf(t, "%d %d", i, j);
				//		//sprintf(t, "%d", j);
				//		string s = t;
				//		putText(Background, s, Point(j*block_width, i*block_height), FONT_HERSHEY_SIMPLEX, 0.25, Scalar(0, 255, 0));
				//	}
				//}
				//if (i == 72 && j == 41){
				//	std::cout << frameNumber << " First_Background" << std::endl << " " << roi_Bg << std::endl;
				//}
				addWeighted(roi_Bg, 0.5, roi_t_Bg, 0.5, 0, roi_Bg);
			}

			bool bgflag = false;
			if (wt_bB[j] == 0 || abs(cur_bI[j] - bB[j]) > threshold_block){

				absdiff(roi_cur, roi_Bg, diff);
				threshold(diff, diff, threshold_pixel, 255, THRESH_TOZERO);
				cvtColor(diff, diff_c1, CV_RGB2GRAY);
				num_pixel = countNonZero(diff_c1);
				if (num_pixel > threshold_num_pixel){				//符合条件的都是前景
					cnt++;
					if (end_fg[j] == frameNumber - 1 && wt_bB[j] != 0){
						wt_bF[j] ++;
						Mat tmp;
						addWeighted(roi_t_Fg, wt_bF[j] / (wt_bF[j] + 1.0), roi_cur, 1.0 / (wt_bF[j] + 1.0), 0, tmp);
						tmp.copyTo(roi_t_Fg);
					} else{
						wt_bF[j] = 0;
						roi_cur.copyTo(roi_t_Fg);
					}
					if (wt_bF[j] >= mfd[j]){
						roi_t_Fg.copyTo(roi_Bg);		//采用替换的方式
						Mat roi_t_Fg_sum;
						cvtColor(roi_t_Fg, roi_t_Fg_sum, CV_RGB2GRAY);

						wt_bB[j] = 0;					//不采用wt_bF[j]，那样导致背景不稳定，来回换
						bB[j] = sum(roi_t_Fg_sum)[0];
						t_bB[j] = bB[j];

						roi_t_Fg = Scalar(0, 0, 0);
						wt_bF[j] = 0;					//防止产生黑点
					}
					diff.copyTo(roi_Fg);
					end_fg[j] = frameNumber;
				} else{			// background	
					bgflag = true;	
				}
			} else{			//background
				bgflag = true;	
			}
			
			if (bgflag == true){
				//随机更新邻近元素背景
				//	0	1	2
				//	3	x	4
				//	5	6	7
				int randPos = rand() % 8;			//产生1-8的随机数
				int randi, randj;
				switch (randPos){
				case 0:	
					randi = i - 1 > 0 ? i - 1 : 0;	
					randj = j - 1 > 0 ? j - 1 : 0;
					break;
				case 1:	
					randi = i - 1 > 0 ? i - 1 : 0;	
					randj = j;
					break;
				case 2:	
					randi = i - 1 > 0 ? i - 1 : 0;	
					randj = j + 1 < num_block_cols ? j + 1 : num_block_cols-1;
					break;
				case 3:	
					randi = i;
					randj = j - 1 > 0 ? j - 1 : 0;
					break;
				case 4:	
					randi = i;
					randj = j + 1 < num_block_cols ? j + 1 : num_block_cols-1;
					break;
				case 5:
					randi = i + 1 < num_block_rows ? i + 1 : num_block_rows-1;
					randj = j - 1 > 0 ? j - 1 : 0;
					break;
				case 6:	
					randi = i + 1 < num_block_rows ? i + 1 : num_block_rows-1;
					randj = j;		
					break;
				case 7:	
					randi = i + 1 < num_block_rows ? i + 1 : num_block_rows-1;
					randj = j + 1 < num_block_cols ? j + 1 : num_block_cols-1;
					break;
				}

				Rect randRect(randj*block_width, randi*block_height, block_width, block_height);
				Mat rand_roi_Bg(Background, randRect);
				Mat rand_roi_t_Fg(tmp_Foreground, randRect);

				int* rand_wt_bB = weight_blockBackground.ptr<int>(randi);
				int* rand_t_bB = tmp_blockBackground.ptr<int>(randi);		//double类型会影响到下一个矩阵
				int* rand_wt_bF = weight_blockForeground.ptr<int>(randi);
				int* rand_bB = blockBackground.ptr<int>(randi);

				if (rand_wt_bF[randj] >= frame_duration*4){
					rand_roi_t_Fg.copyTo(rand_roi_Bg);		//采用替换的方式
					Mat rand_roi_t_Fg_sum;
					cvtColor(rand_roi_t_Fg, rand_roi_t_Fg_sum, CV_RGB2GRAY);

					rand_wt_bB[randj] = 0;					//不采用wt_bF[j]，那样导致背景不稳定，来回换
					rand_bB[randj] = sum(rand_roi_t_Fg_sum)[0];
					rand_t_bB[randj] = rand_bB[randj];

					rand_roi_t_Fg = Scalar(0, 0, 0);
					rand_wt_bF[randj] = 0;					//防止产生黑点
				}
			}//bgflag

		}//j
	}//i

	if (cnt >= (num_block_cols*num_block_rows / 4.0)){		//设置I帧
		ResetModel();
	}
}

void BlockbasedBGS::ResetModel(void){
	blockBackground = cur_blockImage.clone();
	tmp_blockBackground = cur_blockImage.clone();
	last_blockImage = cur_blockImage.clone();
	tmp_Background = cur_RGBImage.clone();
	Background = cur_RGBImage.clone();

	weight_blockBackground = Mat(block_img_dims, CV_32SC1, Scalar(0));
	weight_blockForeground = Mat(block_img_dims, CV_32SC1, Scalar(0));
	end_blockForeground = Mat(block_img_dims, CV_32SC1, Scalar(0));
	flag_blockinitBackground = Mat(block_img_dims, CV_32SC1, Scalar(0));
	mframe_duration = Mat(block_img_dims, CV_32SC1, Scalar(frame_duration));

	tmp_Foreground = Mat(cur_RGBImage.size(), cur_RGBImage.type(), Scalar(0, 0, 0));

	// num of modes for per pixel
	num_pixel_modes = Mat(block_img_dims, CV_32SC1, Scalar(0));
	// GMM for each pixel
	for (unsigned int i = 0; i < num_block_cols*num_block_rows*max_modes; ++i)
	{
		m_modes[i].weight = 0;
		m_modes[i].variance = 0;
		m_modes[i].pixel = 0;
		m_modes[i].significants = 0;
	}

}

void BlockbasedBGS::GetBackground(Mat& img_background)
{
	img_background = Background.clone();
}

void BlockbasedBGS::GetForeground(Mat& img_foreground)
{
	img_foreground = Foreground.clone();
}

void BlockbasedBGS::ShowHeatMap()
{
	imshow("heatMap", heatMap);
}

void BlockbasedBGS::ResetLastBlockImage(const Mat& block_img)
{
	last_blockImage = block_img.clone();
}

void BlockbasedBGS::saveForegroundImg(int _frameNumber)
{
	if (frameNumber == 1){
		system("mkdir foreground");
	}
	string fgImgPath = "./foreground";
	string frameNumber = std::to_string(_frameNumber+1);
	fgImgPath.append("/").append(frameNumber);
	string edgeImgPath = fgImgPath;
	Mat fg = Foreground.clone();
	Mat gray_fg, threshold_fg, morphology_fg;
	cvtColor(fg, gray_fg, CV_RGB2GRAY);
	threshold(gray_fg, threshold_fg, 10, 255, THRESH_BINARY);

	//morphologyEx(threshold_fg, morphology_fg, MORPH_OPEN, Mat(3, 3, CV_8U), Point(-1, -1), 1);
	morphologyEx(threshold_fg, morphology_fg, MORPH_CLOSE, Mat(3, 3, CV_8U), Point(-1, -1), 1);
	morphologyEx(morphology_fg, morphology_fg, MORPH_OPEN, Mat(3, 3, CV_8U), Point(-1, -1), 1);
	imshow("Foreground", morphology_fg);

	imwrite(fgImgPath.append(".png").c_str(), morphology_fg);
	std::cout << "The frame number is " << _frameNumber << std::endl;
	//imwrite(edgeImgPath.append("edge.png").c_str(), img_edge);
}

void BlockbasedBGS::GetEdgeImg(Mat& gray_frame)
{
	int lowThreshold = 70, ratio = 3, kernel_size = 3;
	Mat detected_edges, blur_img;
	blur(gray_frame, blur_img, Size(3, 3));
	//Canny(blur_img, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
	//img_edge = Scalar::all(0);
	//gray_frame.copyTo(img_edge, detected_edges);
	Canny(blur_img, img_edge, lowThreshold, lowThreshold*ratio, kernel_size);
	threshold(img_edge, img_edge, 10, 127, THRESH_BINARY_INV);
	imshow("EdgeInro", img_edge);
}

void BlockbasedBGS::preprocessForegroundImg(int _frameNumber)
{
	system("mkdir foreground");
	string fgImgPath = "./foreground";
	string frameNumber = std::to_string(_frameNumber + 1);
	fgImgPath.append("/").append(frameNumber);
	string edgeImgPath = fgImgPath;
	Mat fg = Foreground.clone();
	Mat gray_fg, threshold_fg, morphology_fg;
	cvtColor(fg, gray_fg, CV_RGB2GRAY);
	threshold(gray_fg, threshold_fg, 10, 254, THRESH_BINARY_INV);
	//threshold_fg.convertTo(threshold_fg, CV_8UC1);

	//for (int i = 0; i < num_block_rows; ++i){
	//	uchar* fg_data = threshold_fg.ptr<uchar>(i);
	//	uchar* edge_data = img_edge.ptr<uchar>(i);
	//	for (int j = 0; j < num_block_cols; ++j){
	//		if (fg_data[j] == edge_data[j]){
	//			fg_data[j] = 0;
	//		}
	//	}
	//}
	Mat foreground_edge = threshold_fg == img_edge;		// 
	absdiff(threshold_fg, foreground_edge, threshold_fg);
	imshow("Foreground", threshold_fg);
	imwrite(fgImgPath.append(".png").c_str(), threshold_fg);
	std::cout << "The frame number is " << _frameNumber << std::endl;
	//imwrite(edgeImgPath.append("edge.png").c_str(), img_edge);
}

int compareBlockGMM(const void* _gmm1, const void* _gmm2)
{
	blockGMM gmm1 = *(blockGMM*)_gmm1;
	blockGMM gmm2 = *(blockGMM*)_gmm2;

	if (gmm1.significants < gmm2.significants)
		return 1;
	else if (gmm1.significants == gmm2.significants)
		return 0;
	else
		return -1;
}

float BlockbasedBGS::GussianVarience(long posPixel, const int& pixel, const float& Alpha, int& numModes)
{
	// calculate distances to the modes (+ sort???)
	// here we need to go in descending order!!!
	long pos;
	bool bFitsPDF = false;

	float fOneMinAlpha = 1 - Alpha;			// 1 - learningrate

	float totalWeight = 0.0f;

	// calculate number of Gaussians to include in the background model
	int backgroundGaussians = 0;
	double sum = 0.0;
	for (int i = 0; i < numModes; ++i)
	{
		//0.75f
		if (sum < m_bg_threshold)
		{
			backgroundGaussians++;
			sum += m_modes[posPixel + i].weight;
		} else
		{
			break;
		}
	}

	// update all distributions and check for match with current pixel
	for (int iModes = 0; iModes < numModes; iModes++)
	{
		pos = posPixel + iModes;
		float weight = m_modes[pos].weight;

		// fit not found yet
		if (!bFitsPDF)
		{
			//check if it belongs to some of the modes
			//calculate distance
			float var = m_modes[pos].variance;
			float mu = m_modes[pos].pixel;

			float distance = mu - pixel;

			// calculate the squared distance
			float dist = distance * distance;

			// a match occurs when the pixel is within sqrt(fTg) standard deviations of the distribution
			if (dist < sigma_para*var)
			{
				bFitsPDF = true;

				//update distribution
				float k = Alpha / weight;		// \rho???
				weight = fOneMinAlpha*weight + Alpha;
				m_modes[pos].weight = weight;
				m_modes[pos].pixel = mu - k*(distance);

				//limit the variance
				float sigmanew = var + k*(dist - var);
				m_modes[pos].variance = sigmanew < 4 ? 4 : sigmanew > 5 * m_variance ? 5 * m_variance : sigmanew;		// 4 < sigmanew < 5*m_variance
				m_modes[pos].significants = m_modes[pos].weight / sqrt(m_modes[pos].variance);						// weight / sigma
			} else  // exceed 3sigma
			{
				weight = fOneMinAlpha*weight;
				if (weight < 0.0)
				{
					weight = 0.0;
					numModes--;
				}
				//update weight and significants
				m_modes[pos].weight = weight;
				m_modes[pos].significants = m_modes[pos].weight / sqrt(m_modes[pos].variance);
			}
		}
		// fit have found
		else 
		{
			weight = fOneMinAlpha*weight;
			if (weight < 0.0)
			{
				weight = 0.0;
				numModes--;
			}
			//update weight and significants
			m_modes[pos].weight = weight;
			m_modes[pos].significants = m_modes[pos].weight / sqrt(m_modes[pos].variance);
		}

		totalWeight += weight;
	}

	// renormalize weights so they add to one
	double invTotalWeight = 1.0 / totalWeight;
	for (int iLocal = 0; iLocal < numModes; iLocal++)
	{
		m_modes[posPixel + iLocal].weight *= (float)invTotalWeight;
		m_modes[posPixel + iLocal].significants = m_modes[posPixel + iLocal].weight / sqrt(m_modes[posPixel + iLocal].variance);
	}

	// Sort significance values so they are in desending order. 
	qsort(&m_modes[posPixel], numModes, sizeof(blockGMM), compareBlockGMM);

	// make new mode if needed and exit
	if (!bFitsPDF)
	{
		if (numModes < max_modes)
		{
			numModes ++;
		} else
		{
			// the weakest mode will be replaced
		}

		pos = posPixel + numModes - 1;

		m_modes[pos].pixel = pixel;
		m_modes[pos].variance = m_variance;
		m_modes[pos].significants = 0;			// will be set below

		if (numModes == 1)
			m_modes[pos].weight = 1;
		else
			m_modes[pos].weight = Alpha;

		//renormalize weights
		int iLocal;
		float sum = 0.0;
		for (iLocal = 0; iLocal < numModes; iLocal++)
		{
			sum += m_modes[posPixel + iLocal].weight;
		}

		double invSum = 1.0 / sum;
		for (iLocal = 0; iLocal < numModes; iLocal++)
		{
			m_modes[posPixel + iLocal].weight *= (float)invSum;
			m_modes[posPixel + iLocal].significants = m_modes[posPixel + iLocal].weight
				/ sqrt(m_modes[posPixel + iLocal].variance);

		}
	}

	// Sort significance values so they are in desending order. 
	qsort(&(m_modes[posPixel]), numModes, sizeof(blockGMM), compareBlockGMM);

	return m_modes[posPixel].variance*sigma_para;
}
