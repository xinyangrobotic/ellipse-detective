#include <cv.h>
#include <highgui.h>

#include "EllipseDetectorYaed.h"
#include <fstream>

float areanum = 0.215;

using namespace std;
using namespace cv;

void OnVideo()
{

	VideoCapture cap(0);
	if(!cap.isOpened()) return;
//	cap.set(CV_CAP_PROP_FRAME_WIDTH,1600);
//	cap.set(CV_CAP_PROP_FRAME_HEIGHT,896);

	int width = 640;
	int height = 480;

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
        vector<coordinate> ellipse_out;
		yaed->DrawDetectedEllipses(resultImage, ellipse_out, ellsYaed);
		cout<<"椭圆数量："<<ellipse_out.size()<<endl;
        vector<int> a;
        Mat gauss, thresh, canny;
        vector< vector<Point> > contours;
        vector< vector<Point> > rects;
		if(ellipse_out.size() == 0){
            namedWindow("Yaed",1);
            imshow("Yaed", resultImage);;
		}
        else {
            for(auto &p:ellipse_out){
			cout<<"x:"<<p.x<<endl
				<<"y:"<<p.y<<endl
				<<"order:"<<(float)p.order<<endl
                <<"a:"<<p.a<<endl;
            cout<<"process"<<endl;
            threshold(gray, thresh, 120, 255, CV_THRESH_BINARY);
            imshow("threshold", thresh);
//            morphologyEx(gauss, gauss, MORPH_CLOSE, (5, 5) );

//			Canny(thresh, canny, 50, 150, 3);
            findContours(thresh, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
            for (int i = 0; i < contours.size(); i++) {
                //拟合出轮廓外侧最小的矩形
                RotatedRect rotate_rect = minAreaRect(contours[i]);
                Point2f *vertices = new Point2f[4];
                rotate_rect.points(vertices);
                if(rotate_rect.size.height < 10 || rotate_rect.size.height > p.a || abs(rotate_rect.center.x - p.x) > 10 || abs(rotate_rect.center.y - p.y) > 10)
                    continue;

                float x12 = (vertices[1].x + vertices[2].x)/2;
                float y12 = (vertices[1].y + vertices[2].y)/2;
                float xt12 = areanum * (rotate_rect.center.x - x12) + x12;
                float yt12 = y12 - areanum * (y12 - rotate_rect.center.y);

                float x30 = (vertices[3].x + vertices[0].x)/2;
                float y30 = (vertices[3].y + vertices[0].y)/2;
                float yt30 = areanum * (rotate_rect.center.y - y30) + y30;
                float xt30 = x30 - areanum * (x30 - rotate_rect.center.x);

                float x23 = (vertices[2].x + vertices[3].x)/2;
                float y23 = (vertices[2].y + vertices[3].y)/2;
                float xt23 = areanum * (rotate_rect.center.x - x23) + x23;
                float yt23 = y23 - areanum * (y23 - rotate_rect.center.y);

                float x01 = (vertices[1].x + vertices[0].x)/2;
                float y01 = (vertices[1].y + vertices[0].y)/2;
                float yt01 = areanum * (rotate_rect.center.y - y01) + y01;
                float xt01 = x01 - areanum * (x01 - rotate_rect.center.x);

                if(abs((gray.at<uchar>(yt12, xt12) - gray.at<uchar>(yt30, xt30))) < 90
				   && abs((gray.at<uchar>(yt23, xt23) - gray.at<uchar>(yt01, xt01))) < 90)
                    cout<<"decide:"<<"T"<<endl;
                else
                    cout<<"decide:"<<"F"<<endl;
                circle(resultImage, Point(rotate_rect.center.x,rotate_rect.center.y), 2,Scalar(255, 255, 0), 1);
                vector<Point> contour;
                for (int i = 0; i < 4; i++) {
                    contour.push_back(vertices[i]);
                }
				vector< vector<Point> > contours;
				contours.push_back(contour);
				drawContours(resultImage, contours, 0, Scalar(255, 255, 0), 1);
                }
            }

		}
		namedWindow("Yaed",1);
		imshow("Yaed", resultImage);


		waitKey(10);
	}
}


int main(int argc, char** argv)
{
	OnVideo();

	return 0;
}

