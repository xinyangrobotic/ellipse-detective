#include <cv.h>
#include <highgui.h>

#include "EllipseDetectorYaed.h"
#include <fstream>

using namespace std;
using namespace cv;

void OnVideo()
{

	VideoCapture cap(0);
	if(!cap.isOpened()) return;
	cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,360);
	cap.set(CAP_PROP_AUTOFOCUS,0);
//	cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
//	cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);
//	double width0=cap.get(CV_CAP_PROP_FRAME_WIDTH);
//	double height0=cap.get(CV_CAP_PROP_FRAME_HEIGHT);
//	cout<<"width0"<<width0<<endl
//						  <<"height0"<<height0<<endl;
	int width = 640;
	int height = 360;

	// Parameters Settings (Sect. 4.2)
	int		iThLength = 16;
	float	fThObb = 3.0f;
	float	fThPos = 1.0f;
	float	fTaoCenters = 0.05f;
	int 	iNs = 16;
	float	fMaxCenterDistance = sqrt(float(width*width + height*height)) * fTaoCenters;

	float	fThScoreScore = 0.4f;

	// Other constant parameters settings.

	// Gaussian filter parameters, in pre-processing
	Size	szPreProcessingGaussKernelSize = Size(5, 5);
	double	dPreProcessingGaussSigma = 1.0;

	float	fDistanceToEllipseContour = 0.1f;	// (Sect. 3.3.1 - Validation)
	float	fMinReliability = 0.4f;	// Const parameters to discard bad ellipses


	// Initialize Detector with selected parameters
	CEllipseDetectorYaed* yaed = new CEllipseDetectorYaed();
	yaed->SetParameters(szPreProcessingGaussKernelSize,
		dPreProcessingGaussSigma,
		fThPos,
		fMaxCenterDistance,
		iThLength,
		fThObb,
		fDistanceToEllipseContour,
		fThScoreScore,
		fMinReliability,
		iNs
		);

	Mat1b gray;
	while(true)
	{

		Mat3b image;
		cap >> image;
		cvtColor(image, gray, COLOR_RGB2GRAY);

		vector<Ellipse> ellsYaed;
		yaed->Detect(gray, ellsYaed);

		vector<double> times = yaed->GetTimes();

		Mat3b resultImage = image.clone();
        vector<coordinate> ellipse_out, ellipse_TF;
		yaed->DrawDetectedEllipses(resultImage, ellipse_out, ellsYaed);
		cout<<"椭圆数量："<<ellipse_out.size()<<endl;
        vector< vector<Point> > contours;
		if(ellipse_out.size() == 0){
		}
        else
			visual_rec(gray, ellipse_out, ellipse_TF, contours);
		for(auto &p:ellipse_TF){
		    cout<<"x:"<<p.x<<endl
                           <<"y"<<p.y<<endl
                                     <<"flag"<<p.flag<<endl;
		}
		    for(auto &p:contours){
				vector< vector<Point> > contours1;
				contours1.push_back(p);
				drawContours(resultImage, contours1, 0, Scalar(255, 255, 0), 1);
		    }
			namedWindow("Yaed", 1);
			imshow("Yaed", resultImage);
			waitKey(10);

	}
}


int main(int argc, char** argv)
{
	OnVideo();

	return 0;
}

