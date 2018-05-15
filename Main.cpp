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
//		for(auto &p:ellipse_out){
//			cout<<"x:"<<p.x<<endl
//				<<"y:"<<p.y<<endl
//				<<"order:"<<(float)p.order<<endl;
//		}
		int num_ellipses = ellsYaed.size();
//		计算中心圆的半径
		float ellipses_a = 0;
		vector<int> a;
		Mat gauss, thresh, canny;
		vector< vector<Point> > contours;
		vector< vector<Point> > rects;

		if(ellsYaed.size() != 0) {
			if(ellsYaed.size() == 1){
				ellipses_a = ellsYaed[0]._a;
			}else
			ellipses_a =
					ellsYaed[num_ellipses - 1]._a < ellsYaed[num_ellipses - 2]._a ? ellsYaed[num_ellipses - 1]._a
																				  : ellsYaed[num_ellipses - 2]._a;
			a.push_back(ellipses_a);
		}else{
			/*
            int j = 0, r = 0;
			for(auto i = a.cend() - 5; i < a.cend(); i++){
				if( (*i) == 0 )
					continue;
				else {
					j = j + 1;
					r = r + (*i);
				}
			}
			ellipses_a = r/j;
			 */
            ellipses_a = 176;
		}
		//将中心圆的半径与全局的rows进行比例换算，若超过1/6，则进行方框判断。
		if (ellipses_a - 175 > 0){
			cout<<"process"<<endl;
//			threshold(gray, thresh, 120, 255, CV_THRESH_BINARY);
//            imshow("threshold", thresh);
//            morphologyEx(gauss, gauss, MORPH_CLOSE, (5, 5) );

//			Canny(thresh, canny, 50, 150, 3);
			findContours(thresh, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
			for (int i = 0; i < contours.size(); i++) {
				//拟合出轮廓外侧最小的矩形
				RotatedRect rotate_rect = minAreaRect(contours[i]);
				Point2f *vertices = new Point2f[4];
				rotate_rect.points(vertices);
                if(rotate_rect.size.height < 10 || rotate_rect.size.height > 500 || gray.at<uchar>(rotate_rect.center.y, rotate_rect.center.x) > 20)
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

                int flag = ((abs(x12 - x30) + abs(y12 - y30)) > (abs(x23 - x01) + abs(y23 - y01)))?12:23;

				float upx, upy, downx, downy;
               	if(flag == 12){
					 upx = xt12, upy = yt12, downx = xt30, downy = yt30;
				} else if(flag == 23)
					 upx = xt23, upy = yt23, downx = xt01, downy = yt01;

                if(abs((gray.at<uchar>(upy, upx) - gray.at<uchar>(downy, downx))) > 90 )
                    cout<<"decide:"<<"F"<<endl;
                else
                    cout<<"decide:"<<"T"<<endl;
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

