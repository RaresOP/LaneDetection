#include "opencv2/highgui.hpp"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "PolyFit.h"
#define M_PI           3.14159265358979323846

using Filter=cv::Mat(*)(cv::Mat);

auto GrayScale(cv::Mat coloredImg)->cv::Mat
{
	cv::Mat grayScaleImg;
	cv::cvtColor(coloredImg, grayScaleImg, cv::COLOR_BGR2GRAY);
	return grayScaleImg;
}

auto GaussianBlr(cv::Mat inputImg)->cv::Mat
{
	cv::Mat output;
	cv::GaussianBlur(inputImg, output, cv::Size(3, 3), 0);
	return output;
}

auto Canny(cv::Mat inputImg)->cv::Mat
{
	cv::Mat output;
	cv::Canny(inputImg,output,170,200);
	return output;
}

auto DrawLines(std::vector<cv::Vec4i> lines, cv::Mat img)->void
{
	std::vector<int> leftLaneX,leftLaneY,rightLaneX,rightLaneY;
	for(auto&& line:lines)
	{
		auto x1=line[0];
		auto y1=line[1];
		auto x2=line[2];
		auto y2=line[3];
		double slope =(double) (y1-y2)/(double)(x1-x2);
		if(slope>0)
		{
			rightLaneX.push_back(x1);
			rightLaneX.push_back(x2);
			rightLaneY.push_back(y1);
			rightLaneY.push_back(y2);
		}
		else
		{
			leftLaneX.push_back(x1);
			leftLaneX.push_back(x2);
			leftLaneY.push_back(y1);
			leftLaneY.push_back(y2);
		}
	}
	if(leftLaneX.empty() || rightLaneX.empty()) return;
        double fitLeft0, fitLeft1, fitRight0, fitRight1;
	PolyFit(leftLaneX, leftLaneY, fitLeft0, fitLeft1);
	PolyFit(rightLaneX, rightLaneY,fitRight0,fitRight1);

	auto YY1 = int(9*img.rows/15);
        auto XX1 = int((YY1 - fitLeft0)/fitLeft1);


        auto YY2 = int(img.rows);
        auto XX2 = int((YY2 - fitLeft0)/fitLeft1);

        if(0<XX1 && XX1<img.cols && 0<XX2 && XX2<img.cols && 0<YY1 && YY1<img.rows && 0<YY2 && YY2<=img.rows)
	{
            cv::line(img, cv::Point(XX1,YY1), cv::Point(XX2,YY2),cv::Scalar(0,0,255), 5);  
	}	


	YY1 = int(9*img.rows/15);
        XX1 = int((YY1 - fitRight0)/fitRight1);

        YY2 = int(img.rows);
        XX2 = int((YY2 - fitRight0)/fitRight1);
        if(0<XX1 && XX1<img.cols && 0<XX2 && XX2<img.cols && 0<YY1 && YY1<img.rows && 0<YY2 && YY2<=img.rows)
	{
            cv::line(img, cv::Point(XX1,YY1),cv::Point (XX2,YY2), cv::Scalar(0,0,255),5);
	}



}

auto Overlay(cv::Mat imgLines, cv::Mat imgInitial)->cv::Mat
{
	cv::Mat output;
	cv::addWeighted(imgLines, 1, imgInitial, 0.8, 0.0, output);
	return output;
}

auto HoughLines(cv::Mat inputImg)->cv::Mat
{
        std::vector<cv::Vec4i> lines;
	
	cv::HoughLinesP(inputImg,lines, 1,M_PI/18,20,9,15 );
	cv::Mat linesImg=cv::Mat::zeros(inputImg.size(),CV_8UC3);
        DrawLines(lines,linesImg);
	return linesImg;
}

auto ApplyFilter(cv::Mat image, Filter filter)->cv::Mat
{
	return filter(image);
}


auto RegionOfInterest(cv::Mat image)->cv::Mat
{
	cv::Point vertices[1][4];
        vertices[0][0] = cv::Point(0, image.rows);
	vertices[0][1] = cv::Point(5*image.cols/11,9*image.rows/15);
	vertices[0][2] = cv::Point(6*image.cols/11,9*image.rows/15);
	vertices[0][3] = cv::Point(image.cols,image.rows);
       
        const cv::Point* vertices_list[1] = {vertices[0]};

	int numberOfPoints=4;
        cv::Mat mask = cv::Mat::zeros(image.size(), image.type());
	cv::fillPoly(mask, vertices_list, &numberOfPoints, 1, cv::Scalar(255), 8);
	cv::Mat output;
	cv::bitwise_and(mask,image,output);
	return output;
}

auto ProcessFrame(cv::Mat frame)->cv::Mat
{
    auto linesImg=frame;
    std::vector<Filter> filters{&GrayScale, &GaussianBlr, &Canny, &RegionOfInterest, &HoughLines};
    for(auto&& currentFilter:filters)
    {
        linesImg=ApplyFilter(linesImg,currentFilter);
    }
    cv::Mat finalFrame=Overlay(linesImg, frame);
    return finalFrame;
}

auto ProcessVideo()->void
{
	cv::VideoCapture cap("../Demo/solidWhiteRight.mp4");
	cv::Mat currentFrame;
	cap>>currentFrame;
	cv::VideoWriter video("outcpp.avi",cv::VideoWriter::fourcc('M','J','P','G'),10,currentFrame.size()); 
	while(!currentFrame.empty())
	{	
	    currentFrame=ProcessFrame(currentFrame);
	    video.write(currentFrame);
	    cap>>currentFrame;
	}
	cap.release();
	video.release();
}

int main( int argc, char** argv ) {
  ProcessVideo();
  return 0;
}
