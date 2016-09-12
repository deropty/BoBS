#include "ZXHBlockbased.h"

using namespace cv;

ZXHBlockbased::ZXHBlockbased() : firstTime(true), frameNumber(0), showOutput(true),
loadDefaultParams(true)
{
	BGS = new BlockbasedBGS();
	std::cout << "ZXHBlockbased()" << std::endl;
}

ZXHBlockbased::~ZXHBlockbased()
{
	//finish();
	std::cout << "~ZXHBlockbased()" << std::endl;
}

//void ZXHBlockbased::setStatus(Status _status)
//{
//	status = _status;
//}

//void ZXHBlockbased::finish(void)
//{
//	delete BGS;
//}

void ZXHBlockbased::process(const Mat &img_input, Mat &img_output, Mat &img_bgmodel)
{
	if (img_input.empty())
		return;

	//loadConfig();

	if (firstTime && loadDefaultParams){
		block_height = 2;			//实现非正方形形状
		block_width = 2;
		frame_duration = 5;
		threshold_pixel = 12;		//2,2,20,12
		block = Size(block_width, block_height);
		img_dims = Size(img_input.cols, img_input.rows);
		int bid_width = img_input.cols / block_width;
		int bid_height = img_input.rows / block_height;
		block_img_dims = Size(bid_width, bid_height);
	}
	
	Mat block_frame(block_img_dims, CV_32SC1);		//方便下面使用高效率的指针，而非at
	Mat gray_frame(img_dims, CV_32SC1);
	preprocess(img_input, block_frame, block, gray_frame);		//得到分块的积分图

	if (firstTime)
	{
		BGS->InitPara(block, img_dims, frame_duration, threshold_pixel);
		BGS->InitModel(img_input, block_frame, frameNumber);
		//saveConfig();
		firstTime = false;
		++frameNumber;
		return;
	}

	BGS->SetRGBInputImage(img_input, gray_frame, frameNumber);
	BGS->SetBlockImage(block_frame);

	BGS->process();

	BGS->ResetLastBlockImage(block_frame);
	//BGS->GetEdgeImg(gray_frame);
	//BGS->ShowHeatMap();


	BGS->GetBackground(img_background);
	BGS->GetForeground(img_foreground);
	//BGS->preprocessForegroundImg(frameNumber);
	BGS->saveForegroundImg(frameNumber);




	//threshold(gray_img_foreground, gray_img_foreground, 8, 255, THRESH_BINARY);


	if (showOutput)
	{
		imshow("Background", img_background);
		//imshow("Foreground", img_foreground);
		
		waitKey(30);
	}

	
	img_foreground.copyTo(img_output);
	img_background.copyTo(img_bgmodel);

	//delete img;
	////cvReleaseImage(&img);

	//firstTime = false;
	frameNumber++;
}

void ZXHBlockbased::saveConfig()
{
	CvFileStorage* fs = cvOpenFileStorage("ZXHBlockbased.xml", 0, CV_STORAGE_WRITE);

	cvWriteInt(fs, "frame_duration", frame_duration);
	cvWriteInt(fs, "block_height", block_height);
	cvWriteInt(fs, "block_width", block_width); 

	cvWriteInt(fs, "showOutput", showOutput);

	cvReleaseFileStorage(&fs);
}

void ZXHBlockbased::loadConfig()
{
	CvFileStorage* fs = cvOpenFileStorage("ZXHBlockbased.xml", 0, CV_STORAGE_READ);

	frame_duration = cvReadIntByName(fs, 0, "frame_duration", 0);
	block_height = cvReadIntByName(fs, 0, "block_height", 0);
	block_width = cvReadIntByName(fs, 0, "block_width", 0);
	
	showOutput = cvReadIntByName(fs, 0, "showOutput", true);

	cvReleaseFileStorage(&fs);
}

void ZXHBlockbased::preprocess(const Mat& img_input, Mat& block_frame, Size block, Mat& grey_img)
{
	Mat integral_img;
	cv::cvtColor(img_input, grey_img, CV_RGB2GRAY);
	cv::integral(grey_img, integral_img);

	for (int P = 0, p = 0; p < block_img_dims.height; ++p, P += block.height){
		int* block_data = block_frame.ptr<int>(p);
		for (int Q = 0, q = 0; q < block_img_dims.width; ++q, Q += block.width){
			block_data[q] = integral_img.at<int>(P + block.height - 1, Q + block.width - 1) + integral_img.at<int>(P, Q)
				- integral_img.at<int>(P + block.height - 1, Q) - integral_img.at<int>(P, Q + block.width - 1);
		}
	}

}

