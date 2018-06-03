/*
This code is intended for academic use only.
You are free to use and modify the code, at your own risk.

If you use this code, or find it useful, please refer to the paper:

Michele Fornaciari, Andrea Prati, Rita Cucchiara, 
A fast and effective ellipse detector for embedded vision applications
Pattern Recognition, Volume 47, Issue 11, November 2014, Pages 3693-3708, ISSN 0031-3203, 
http://dx.doi.org/10.1016/j.patcog.2014.05.012.
(http://www.sciencedirect.com/science/article/pii/S0031320314001976)


The comments in the code refer to the abovementioned paper.
If you need further details about the code or the algorithm, please contact me at:

michele.fornaciari@unimore.it

last update: 23/12/2014
*/

#include "EllipseDetectorYaed.h"


CEllipseDetectorYaed::CEllipseDetectorYaed(void) : _times(6, 0.0), _timesHelper(6, 0.0)
{
	// Default Parameters Settings
	_szPreProcessingGaussKernelSize = Size(5, 5);
	_dPreProcessingGaussSigma = 1.0;
	_fThPosition = 1.0f;
	_fMaxCenterDistance = 100.0f * 0.05f;
	_fMaxCenterDistance2 = _fMaxCenterDistance * _fMaxCenterDistance;
	_iMinEdgeLength = 16;
	_fMinOrientedRectSide = 3.0f;
	_fDistanceToEllipseContour = 0.1f;
	_fMinScore = 0.4f;
	_fMinReliability = 0.4f;
	_uNs = 16;

	srand(unsigned(time(NULL)));
}


CEllipseDetectorYaed::~CEllipseDetectorYaed(void)
{
}

void CEllipseDetectorYaed::SetParameters(Size	szPreProcessingGaussKernelSize,
	double	dPreProcessingGaussSigma,
	float 	fThPosition,
	float	fMaxCenterDistance,
	int		iMinEdgeLength,
	float	fMinOrientedRectSide,
	float	fDistanceToEllipseContour,
	float	fMinScore,
	float	fMinReliability,
	int     iNs
	)
{
	_szPreProcessingGaussKernelSize = szPreProcessingGaussKernelSize;
	_dPreProcessingGaussSigma = dPreProcessingGaussSigma;
	_fThPosition = fThPosition;
	_fMaxCenterDistance = fMaxCenterDistance;
	_iMinEdgeLength = iMinEdgeLength;
	_fMinOrientedRectSide = fMinOrientedRectSide;
	_fDistanceToEllipseContour = fDistanceToEllipseContour;
	_fMinScore = fMinScore;
	_fMinReliability = fMinReliability;
	_uNs = iNs;

	_fMaxCenterDistance2 = _fMaxCenterDistance * _fMaxCenterDistance;

}

uint inline CEllipseDetectorYaed::GenerateKey(uchar pair, ushort u, ushort v)
{
	return (pair << 30) + (u << 15) + v;
};



int CEllipseDetectorYaed::FindMaxK(const int* v) const
{
	int max_val = 0;
	int max_idx = 0;
	for (int i = 0; i<ACC_R_SIZE; ++i)
	{
		(v[i] > max_val) ? max_val = v[i], max_idx = i : 0;
	}

	return max_idx + 90;
};

int CEllipseDetectorYaed::FindMaxN(const int* v) const
{
	int max_val = 0;
	int max_idx = 0;
	for (int i = 0; i<ACC_N_SIZE; ++i)
	{
		(v[i] > max_val) ? max_val = v[i], max_idx = i : 0;
	}

	return max_idx;
};

int CEllipseDetectorYaed::FindMaxA(const int* v) const
{
	int max_val = 0;
	int max_idx = 0;
	for (int i = 0; i<ACC_A_SIZE; ++i)
	{
		(v[i] > max_val) ? max_val = v[i], max_idx = i : 0;
	}

	return max_idx;
};


float CEllipseDetectorYaed::GetMedianSlope(vector<Point2f>& med, Point2f& M, vector<float>& slopes)
{
	// med		: vector of points
	// M		: centroid of the points in med
	// slopes	: vector of the slopes

	unsigned iNofPoints = med.size();
	//CV_Assert(iNofPoints >= 2);

	unsigned halfSize = iNofPoints >> 1;
	unsigned quarterSize = halfSize >> 1;

	vector<float> xx, yy;
	slopes.reserve(halfSize);
	xx.reserve(iNofPoints);
	yy.reserve(iNofPoints);

	for (unsigned i = 0; i < halfSize; ++i)
	{
		Point2f& p1 = med[i];
		Point2f& p2 = med[halfSize + i];

		xx.push_back(p1.x);
		xx.push_back(p2.x);
		yy.push_back(p1.y);
		yy.push_back(p2.y);

		float den = (p2.x - p1.x);
		float num = (p2.y - p1.y);

		if (den == 0) den = 0.00001f;

		slopes.push_back(num / den);
	}

	nth_element(slopes.begin(), slopes.begin() + quarterSize, slopes.end());
	nth_element(xx.begin(), xx.begin() + halfSize, xx.end());
	nth_element(yy.begin(), yy.begin() + halfSize, yy.end());
	M.x = xx[halfSize];
	M.y = yy[halfSize];

	return slopes[quarterSize];
};




void CEllipseDetectorYaed::GetFastCenter(vector<Point>& e1, vector<Point>& e2, EllipseData& data)
{
	data.isValid = true;

	unsigned size_1 = unsigned(e1.size());
	unsigned size_2 = unsigned(e2.size());

	unsigned hsize_1 = size_1 >> 1;
	unsigned hsize_2 = size_2 >> 1;

	Point& med1 = e1[hsize_1];
	Point& med2 = e2[hsize_2];

	Point2f M12, M34;
	float q2, q4;

	{
		// First to second

		// Reference slope

		float dx_ref = float(e1[0].x - med2.x);
		float dy_ref = float(e1[0].y - med2.y);

		if (dy_ref == 0) dy_ref = 0.00001f;

		float m_ref = dy_ref / dx_ref;
		data.ra = m_ref;

		// Find points with same slope as reference
		vector<Point2f> med;
		med.reserve(hsize_2);

		unsigned minPoints = (_uNs < hsize_2) ? _uNs : hsize_2;

		vector<uint> indexes(minPoints);
		if (_uNs < hsize_2)
		{
			unsigned iSzBin = hsize_2 / unsigned(_uNs);
			unsigned iIdx = hsize_2 + (iSzBin / 2);

			for (unsigned i = 0; i<_uNs; ++i)
			{
				indexes[i] = iIdx;
				iIdx += iSzBin;
			}
		}
		else
		{
			iota(indexes.begin(), indexes.end(), hsize_2);
		}



		for (uint ii = 0; ii<minPoints; ++ii)
		{
			uint i = indexes[ii];

			float x1 = float(e2[i].x);
			float y1 = float(e2[i].y);

			uint begin = 0;
			uint end = size_1 - 1;

			float xb = float(e1[begin].x);
			float yb = float(e1[begin].y);
			float res_begin = ((xb - x1) * dy_ref) - ((yb - y1) * dx_ref);
			int sign_begin = sgn(res_begin);
			if (sign_begin == 0)
			{
				//found
				med.push_back(Point2f((xb + x1)* 0.5f, (yb + y1)* 0.5f));
				continue;
			}

			float xe = float(e1[end].x);
			float ye = float(e1[end].y);
			float res_end = ((xe - x1) * dy_ref) - ((ye - y1) * dx_ref);
			int sign_end = sgn(res_end);
			if (sign_end == 0)
			{
				//found
				med.push_back(Point2f((xe + x1)* 0.5f, (ye + y1)* 0.5f));
				continue;
			}

			if ((sign_begin + sign_end) != 0)
			{
				continue;
			}

			uint j = (begin + end) >> 1;

			while (end - begin > 2)
			{
				float x2 = float(e1[j].x);
				float y2 = float(e1[j].y);
				float res = ((x2 - x1) * dy_ref) - ((y2 - y1) * dx_ref);
				int sign_res = sgn(res);

				if (sign_res == 0)
				{
					//found
					med.push_back(Point2f((x2 + x1)* 0.5f, (y2 + y1)* 0.5f));
					break;
				}

				if (sign_res + sign_begin == 0)
				{
					sign_end = sign_res;
					end = j;
				}
				else
				{
					sign_begin = sign_res;
					begin = j;
				}
				j = (begin + end) >> 1;
			}

			med.push_back(Point2f((e1[j].x + x1)* 0.5f, (e1[j].y + y1)* 0.5f));
		}

		if (med.size() < 2)
		{
			data.isValid = false;
			return;
		}

		q2 = GetMedianSlope(med, M12, data.Sa);
	}

	{
		// Second to first

		// Reference slope
		float dx_ref = float(med1.x - e2[0].x);
		float dy_ref = float(med1.y - e2[0].y);

		if (dy_ref == 0) dy_ref = 0.00001f;

		float m_ref = dy_ref / dx_ref;
		data.rb = m_ref;

		// Find points with same slope as reference
		vector<Point2f> med;
		med.reserve(hsize_1);

		uint minPoints = (_uNs < hsize_1) ? _uNs : hsize_1;

		vector<uint> indexes(minPoints);
		if (_uNs < hsize_1)
		{
			unsigned iSzBin = hsize_1 / unsigned(_uNs);
			unsigned iIdx = hsize_1 + (iSzBin / 2);

			for (unsigned i = 0; i<_uNs; ++i)
			{
				indexes[i] = iIdx;
				iIdx += iSzBin;
			}
		}
		else
		{
			iota(indexes.begin(), indexes.end(), hsize_1);
		}


		for (uint ii = 0; ii<minPoints; ++ii)
		{
			uint i = indexes[ii];

			float x1 = float(e1[i].x);
			float y1 = float(e1[i].y);

			uint begin = 0;
			uint end = size_2 - 1;

			float xb = float(e2[begin].x);
			float yb = float(e2[begin].y);
			float res_begin = ((xb - x1) * dy_ref) - ((yb - y1) * dx_ref);
			int sign_begin = sgn(res_begin);
			if (sign_begin == 0)
			{
				//found
				med.push_back(Point2f((xb + x1)* 0.5f, (yb + y1)* 0.5f));
				continue;
			}

			float xe = float(e2[end].x);
			float ye = float(e2[end].y);
			float res_end = ((xe - x1) * dy_ref) - ((ye - y1) * dx_ref);
			int sign_end = sgn(res_end);
			if (sign_end == 0)
			{
				//found
				med.push_back(Point2f((xe + x1)* 0.5f, (ye + y1)* 0.5f));
				continue;
			}

			if ((sign_begin + sign_end) != 0)
			{
				continue;
			}

			uint j = (begin + end) >> 1;

			while (end - begin > 2)
			{
				float x2 = float(e2[j].x);
				float y2 = float(e2[j].y);
				float res = ((x2 - x1) * dy_ref) - ((y2 - y1) * dx_ref);
				int sign_res = sgn(res);

				if (sign_res == 0)
				{
					//found
					med.push_back(Point2f((x2 + x1)* 0.5f, (y2 + y1)* 0.5f));
					break;
				}

				if (sign_res + sign_begin == 0)
				{
					sign_end = sign_res;
					end = j;
				}
				else
				{
					sign_begin = sign_res;
					begin = j;
				}
				j = (begin + end) >> 1;
			}

			med.push_back(Point2f((e2[j].x + x1)* 0.5f, (e2[j].y + y1)* 0.5f));
		}
		
		if (med.size() < 2)
		{
			data.isValid = false;
			return;
		}
		q4 = GetMedianSlope(med, M34, data.Sb);
	}

	if (q2 == q4)
	{
		data.isValid = false;
		return;
	}

	float invDen = 1 / (q2 - q4);
	data.Cab.x = (M34.y - q4*M34.x - M12.y + q2*M12.x) * invDen;
	data.Cab.y = (q2*M34.y - q4*M12.y + q2*q4*(M12.x - M34.x)) * invDen;
	data.ta = q2;
	data.tb = q4;
	data.Ma = M12;
	data.Mb = M34;
};




void CEllipseDetectorYaed::DetectEdges13(Mat1b& DP, VVP& points_1, VVP& points_3)
{
	// Vector of connected edge points
	VVP contours;

	// Labeling 8-connected edge points, discarding edge too small
	Labeling(DP, contours, _iMinEdgeLength);
	int iContoursSize = int(contours.size());

	// For each edge
	for (int i = 0; i < iContoursSize; ++i)
	{
		VP& edgeSegment = contours[i];

#ifndef DISCARD_CONSTRAINT_OBOX

		// Selection strategy - Step 1 - See Sect [3.1.2] of the paper
		// Constraint on axes aspect ratio
		RotatedRect oriented = minAreaRect(edgeSegment);
		float o_min = min(oriented.size.width, oriented.size.height);

		if (o_min < _fMinOrientedRectSide)
		{
			continue;
		}
#endif

		// Order edge points of the same arc
		sort(edgeSegment.begin(), edgeSegment.end(), SortTopLeft2BottomRight);
		int iEdgeSegmentSize = unsigned(edgeSegment.size());

		// Get extrema of the arc
		Point& left = edgeSegment[0];
		Point& right = edgeSegment[iEdgeSegmentSize - 1];

		// Find convexity - See Sect [3.1.3] of the paper
		int iCountTop = 0;
		int xx = left.x;
		for (int k = 1; k < iEdgeSegmentSize; ++k)
		{
			if (edgeSegment[k].x == xx) continue;

			iCountTop += (edgeSegment[k].y - left.y);
			xx = edgeSegment[k].x;
		}

		int width = abs(right.x - left.x) + 1;
		int height = abs(right.y - left.y) + 1;
		int iCountBottom = (width * height) - iEdgeSegmentSize - iCountTop;

		if (iCountBottom > iCountTop)
		{	//1
			points_1.push_back(edgeSegment);
		}
		else if (iCountBottom < iCountTop)
		{	//3
			points_3.push_back(edgeSegment);
		}
	}
};


void CEllipseDetectorYaed::DetectEdges24(Mat1b& DN, VVP& points_2, VVP& points_4 )
{
	// Vector of connected edge points
	VVP contours;

	/// Labeling 8-connected edge points, discarding edge too small
	Labeling(DN, contours, _iMinEdgeLength);

	int iContoursSize = unsigned(contours.size());

	// For each edge
	for (int i = 0; i < iContoursSize; ++i)
	{
		VP& edgeSegment = contours[i];


#ifndef DISCARD_CONSTRAINT_OBOX

		// Selection strategy - Step 1 - See Sect [3.1.2] of the paper
		// Constraint on axes aspect ratio
		RotatedRect oriented = minAreaRect(edgeSegment);
		float o_min = min(oriented.size.width, oriented.size.height);

		if (o_min < _fMinOrientedRectSide)
		{
			continue;
		}

#endif

		// Order edge points of the same arc
		sort(edgeSegment.begin(), edgeSegment.end(), SortBottomLeft2TopRight);
		int iEdgeSegmentSize = unsigned(edgeSegment.size());

		// Get extrema of the arc
		Point& left = edgeSegment[0];
		Point& right = edgeSegment[iEdgeSegmentSize - 1];

		// Find convexity - See Sect [3.1.3] of the paper
		int iCountBottom = 0;
		int xx = left.x;
		for (int k = 1; k < iEdgeSegmentSize; ++k)
		{
			if (edgeSegment[k].x == xx) continue;

			iCountBottom += (left.y - edgeSegment[k].y);
			xx = edgeSegment[k].x;
		}

		int width = abs(right.x - left.x) + 1;
		int height = abs(right.y - left.y) + 1;
		int iCountTop = (width *height) - iEdgeSegmentSize - iCountBottom;

		if (iCountBottom > iCountTop)
		{
			//2
			points_2.push_back(edgeSegment);
		}
		else if (iCountBottom < iCountTop)
		{
			//4
			points_4.push_back(edgeSegment);
		}
	}
};

// Most important function for detecting ellipses. See Sect[3.2.3] of the paper
void CEllipseDetectorYaed::FindEllipses(	Point2f& center,
											VP& edge_i,
											VP& edge_j,
											VP& edge_k,
											EllipseData& data_ij,
											EllipseData& data_ik,
											vector<Ellipse>& ellipses
										)
{
	// Find ellipse parameters

	// 0-initialize accumulators
	memset(accN, 0, sizeof(int)*ACC_N_SIZE);
	memset(accR, 0, sizeof(int)*ACC_R_SIZE);
	memset(accA, 0, sizeof(int)*ACC_A_SIZE);

	Tac(3); //estimation

	// Get size of the 4 vectors of slopes (2 pairs of arcs)
	int sz_ij1 = int(data_ij.Sa.size());
	int sz_ij2 = int(data_ij.Sb.size());
	int sz_ik1 = int(data_ik.Sa.size());
	int sz_ik2 = int(data_ik.Sb.size());

	// Get the size of the 3 arcs
	size_t sz_ei = edge_i.size();
	size_t sz_ej = edge_j.size();
	size_t sz_ek = edge_k.size();

	// Center of the estimated ellipse
	float a0 = center.x;
	float b0 = center.y;


	// Estimation of remaining parameters
	// Uses 4 combinations of parameters. See Table 1 and Sect [3.2.3] of the paper.
	{
		float q1 = data_ij.ra;
		float q3 = data_ik.ra;
		float q5 = data_ik.rb;

		for (int ij1 = 0; ij1 < sz_ij1; ++ij1)
		{
			float q2 = data_ij.Sa[ij1];

			float q1xq2 = q1*q2;

			for (int ik1 = 0; ik1 < sz_ik1; ++ik1)
			{
				float q4 = data_ik.Sa[ik1];

				float q3xq4 = q3*q4;

				// See Eq. [13-18] in the paper

				float a = (q1xq2 - q3xq4);
				float b = (q3xq4 + 1)*(q1 + q2) - (q1xq2 + 1)*(q3 + q4);
				float Kp = (-b + sqrt(b*b + 4 * a*a)) / (2 * a);
				float zplus = ((q1 - Kp)*(q2 - Kp)) / ((1 + q1*Kp)*(1 + q2*Kp));

				if (zplus >= 0.0f)
				{
					continue;
				}

				float Np = sqrt(-zplus);
				float rho = atan(Kp);
				int rhoDeg;
				if (Np > 1.f)
				{
					Np = 1.f / Np;
					rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180; // [0,180)					
				}
				else
				{
					rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180; // [0,180)
				}

				int iNp = cvRound(Np * 100); // [0, 100]

				if (0 <= iNp	&& iNp < ACC_N_SIZE &&
					0 <= rhoDeg	&& rhoDeg < ACC_R_SIZE
					)
				{
					++accN[iNp];	// Increment N accumulator
					++accR[rhoDeg];	// Increment R accumulator
				}
			}


			for (int ik2 = 0; ik2 < sz_ik2; ++ik2)
			{
				float q4 = data_ik.Sb[ik2];

				float q5xq4 = q5*q4;

				// See Eq. [13-18] in the paper

				float a = (q1xq2 - q5xq4);
				float b = (q5xq4 + 1)*(q1 + q2) - (q1xq2 + 1)*(q5 + q4);
				float Kp = (-b + sqrt(b*b + 4 * a*a)) / (2 * a);
				float zplus = ((q1 - Kp)*(q2 - Kp)) / ((1 + q1*Kp)*(1 + q2*Kp));

				if (zplus >= 0.0f)
				{
					continue;
				}

				float Np = sqrt(-zplus);
				float rho = atan(Kp);
				int rhoDeg;
				if (Np > 1.f)
				{
					Np = 1.f / Np;
					rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180; // [0,180)					
				}
				else
				{
					rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180; // [0,180)
				}

				int iNp = cvRound(Np * 100); // [0, 100]

				if (0 <= iNp	&& iNp < ACC_N_SIZE &&
					0 <= rhoDeg	&& rhoDeg < ACC_R_SIZE
					)
				{
					++accN[iNp];		// Increment N accumulator
					++accR[rhoDeg];		// Increment R accumulator
				}
			}

		}
	}


	{
		float q1 = data_ij.rb;
		float q3 = data_ik.rb;
		float q5 = data_ik.ra;

		for (int ij2 = 0; ij2 < sz_ij2; ++ij2)
		{
			float q2 = data_ij.Sb[ij2];

			float q1xq2 = q1*q2;

			for (int ik2 = 0; ik2 < sz_ik2; ++ik2)
			{
				float q4 = data_ik.Sb[ik2];

				float q3xq4 = q3*q4;

				// See Eq. [13-18] in the paper

				float a = (q1xq2 - q3xq4);
				float b = (q3xq4 + 1)*(q1 + q2) - (q1xq2 + 1)*(q3 + q4);
				float Kp = (-b + sqrt(b*b + 4 * a*a)) / (2 * a);
				float zplus = ((q1 - Kp)*(q2 - Kp)) / ((1 + q1*Kp)*(1 + q2*Kp));

				if (zplus >= 0.0f)
				{
					continue;
				}

				float Np = sqrt(-zplus);
				float rho = atan(Kp);
				int rhoDeg;
				if (Np > 1.f)
				{
					Np = 1.f / Np;
					rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180; // [0,180)
				}
				else
				{
					rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180; // [0,180)
				}

				int iNp = cvRound(Np * 100); // [0, 100]

				if (0 <= iNp	&& iNp < ACC_N_SIZE &&
					0 <= rhoDeg	&& rhoDeg < ACC_R_SIZE
					)
				{
					++accN[iNp];		// Increment N accumulator
					++accR[rhoDeg];		// Increment R accumulator
				}
			}


			for (int ik1 = 0; ik1 < sz_ik1; ++ik1)
			{
				float q4 = data_ik.Sa[ik1];

				float q5xq4 = q5*q4;

				// See Eq. [13-18] in the paper

				float a = (q1xq2 - q5xq4);
				float b = (q5xq4 + 1)*(q1 + q2) - (q1xq2 + 1)*(q5 + q4);
				float Kp = (-b + sqrt(b*b + 4 * a*a)) / (2 * a);
				float zplus = ((q1 - Kp)*(q2 - Kp)) / ((1 + q1*Kp)*(1 + q2*Kp));

				if (zplus >= 0.0f)
				{
					continue;
				}

				float Np = sqrt(-zplus);
				float rho = atan(Kp);
				int rhoDeg;
				if (Np > 1.f)
				{
					Np = 1.f / Np;
					rhoDeg = cvRound((rho * 180 / CV_PI) + 180) % 180; // [0,180)
				}
				else
				{
					rhoDeg = cvRound((rho * 180 / CV_PI) + 90) % 180; // [0,180)
				}

				int iNp = cvRound(Np * 100); // [0, 100]

				if (0 <= iNp	&& iNp < ACC_N_SIZE &&
					0 <= rhoDeg	&& rhoDeg < ACC_R_SIZE
					)
				{
					++accN[iNp];		// Increment N accumulator
					++accR[rhoDeg];		// Increment R accumulator
				}
			}

		}
	}

	// Find peak in N and K accumulator
	int iN = FindMaxN(accN);
	int iK = FindMaxK(accR);

	// Recover real values
	float fK = float(iK);
	float Np = float(iN) * 0.01f;
	float rho = fK * float(CV_PI) / 180.f;	//deg 2 rad
	float Kp = tan(rho);

	// Estimate A. See Eq. [19 - 22] in Sect [3.2.3] of the paper

	for (ushort l = 0; l < sz_ei; ++l)
	{
		Point& pp = edge_i[l];
		float sk = 1.f / sqrt(Kp*Kp + 1.f);
		float x0 = ((pp.x - a0) * sk) + (((pp.y - b0)*Kp) * sk);
		float y0 = -(((pp.x - a0) * Kp) * sk) + ((pp.y - b0) * sk);
		float Ax = sqrt((x0*x0*Np*Np + y0*y0) / ((Np*Np)*(1.f + Kp*Kp)));
		int A = cvRound(abs(Ax / cos(rho)));
		if ((0 <= A) && (A < ACC_A_SIZE))
		{
			++accA[A];
		}
	}

	for (ushort l = 0; l < sz_ej; ++l)
	{
		Point& pp = edge_j[l];
		float sk = 1.f / sqrt(Kp*Kp + 1.f);
		float x0 = ((pp.x - a0) * sk) + (((pp.y - b0)*Kp) * sk);
		float y0 = -(((pp.x - a0) * Kp) * sk) + ((pp.y - b0) * sk);
		float Ax = sqrt((x0*x0*Np*Np + y0*y0) / ((Np*Np)*(1.f + Kp*Kp)));
		int A = cvRound(abs(Ax / cos(rho)));
		if ((0 <= A) && (A < ACC_A_SIZE))
		{
			++accA[A];
		}
	}

	for (ushort l = 0; l < sz_ek; ++l)
	{
		Point& pp = edge_k[l];
		float sk = 1.f / sqrt(Kp*Kp + 1.f);
		float x0 = ((pp.x - a0) * sk) + (((pp.y - b0)*Kp) * sk);
		float y0 = -(((pp.x - a0) * Kp) * sk) + ((pp.y - b0) * sk);
		float Ax = sqrt((x0*x0*Np*Np + y0*y0) / ((Np*Np)*(1.f + Kp*Kp)));
		int A = cvRound(abs(Ax / cos(rho)));
		if ((0 <= A) && (A < ACC_A_SIZE))
		{
			++accA[A];
		}
	}

	// Find peak in A accumulator
	int A = FindMaxA(accA);
	float fA = float(A);

	// Find B value. See Eq [23] in the paper
	float fB = abs(fA * Np);

	// Got all ellipse parameters!
	Ellipse ell(a0, b0, fA, fB, fmod(rho + float(CV_PI)*2.f, float(CV_PI)));

	Toc(3); //estimation
	Tac(4); //validation

	// Get the score. See Sect [3.3.1] in the paper

	// Find the number of edge pixel lying on the ellipse
	float _cos = cos(-ell._rad);
	float _sin = sin(-ell._rad);

	float invA2 = 1.f / (ell._a * ell._a);
	float invB2 = 1.f / (ell._b * ell._b);

	float invNofPoints = 1.f / float(sz_ei + sz_ej + sz_ek);
	int counter_on_perimeter = 0;

	for (ushort l = 0; l < sz_ei; ++l)
	{
		float tx = float(edge_i[l].x) - ell._xc;
		float ty = float(edge_i[l].y) - ell._yc;
		float rx = (tx*_cos - ty*_sin);
		float ry = (tx*_sin + ty*_cos);

		float h = (rx*rx)*invA2 + (ry*ry)*invB2;
		if (abs(h - 1.f) < _fDistanceToEllipseContour)
		{
			++counter_on_perimeter;
		}
	}

	for (ushort l = 0; l < sz_ej; ++l)
	{
		float tx = float(edge_j[l].x) - ell._xc;
		float ty = float(edge_j[l].y) - ell._yc;
		float rx = (tx*_cos - ty*_sin);
		float ry = (tx*_sin + ty*_cos);

		float h = (rx*rx)*invA2 + (ry*ry)*invB2;
		if (abs(h - 1.f) < _fDistanceToEllipseContour)
		{
			++counter_on_perimeter;
		}
	}

	for (ushort l = 0; l < sz_ek; ++l)
	{
		float tx = float(edge_k[l].x) - ell._xc;
		float ty = float(edge_k[l].y) - ell._yc;
		float rx = (tx*_cos - ty*_sin);
		float ry = (tx*_sin + ty*_cos);

		float h = (rx*rx)*invA2 + (ry*ry)*invB2;
		if (abs(h - 1.f) < _fDistanceToEllipseContour)
		{
			++counter_on_perimeter;
		}
	}

	//no points found on the ellipse
	if (counter_on_perimeter <= 0)
	{
		Toc(4); //validation
		return;
	}

	// Compute score
	float score = float(counter_on_perimeter) * invNofPoints;
	if (score < _fMinScore)
	{
		Toc(4); //validation
		return;
	}

	// Compute reliability	
	// this metric is not described in the paper, mostly due to space limitations.
	// The main idea is that for a given ellipse (TD) even if the score is high, the arcs 
	// can cover only a small amount of the contour of the estimated ellipse. 
	// A low reliability indicate that the arcs form an elliptic shape by chance, but do not underlie
	// an actual ellipse. The value is normalized between 0 and 1. 
	// The default value is 0.4.

	// It is somehow similar to the "Angular Circumreference Ratio" saliency criteria 
	// as in the paper: 
	// D. K. Prasad, M. K. Leung, S.-Y. Cho, Edge curvature and convexity
	// based ellipse detection method, Pattern Recognition 45 (2012) 3204-3221.

	float di, dj, dk;
	{
		Point2f p1(float(edge_i[0].x), float(edge_i[0].y));
		Point2f p2(float(edge_i[sz_ei - 1].x), float(edge_i[sz_ei - 1].y));
		p1.x -= ell._xc;
		p1.y -= ell._yc;
		p2.x -= ell._xc;
		p2.y -= ell._yc;
		Point2f r1((p1.x*_cos - p1.y*_sin), (p1.x*_sin + p1.y*_cos));
		Point2f r2((p2.x*_cos - p2.y*_sin), (p2.x*_sin + p2.y*_cos));
		di = abs(r2.x - r1.x) + abs(r2.y - r1.y);
	}
	{
		Point2f p1(float(edge_j[0].x), float(edge_j[0].y));
		Point2f p2(float(edge_j[sz_ej - 1].x), float(edge_j[sz_ej - 1].y));
		p1.x -= ell._xc;
		p1.y -= ell._yc;
		p2.x -= ell._xc;
		p2.y -= ell._yc;
		Point2f r1((p1.x*_cos - p1.y*_sin), (p1.x*_sin + p1.y*_cos));
		Point2f r2((p2.x*_cos - p2.y*_sin), (p2.x*_sin + p2.y*_cos));
		dj = abs(r2.x - r1.x) + abs(r2.y - r1.y);
	}
	{
		Point2f p1(float(edge_k[0].x), float(edge_k[0].y));
		Point2f p2(float(edge_k[sz_ek - 1].x), float(edge_k[sz_ek - 1].y));
		p1.x -= ell._xc;
		p1.y -= ell._yc;
		p2.x -= ell._xc;
		p2.y -= ell._yc;
		Point2f r1((p1.x*_cos - p1.y*_sin), (p1.x*_sin + p1.y*_cos));
		Point2f r2((p2.x*_cos - p2.y*_sin), (p2.x*_sin + p2.y*_cos));
		dk = abs(r2.x - r1.x) + abs(r2.y - r1.y);
	}

	// This allows to get rid of thick edges
	float rel = min(1.f, ((di + dj + dk) / (3 * (ell._a + ell._b))));

	if (rel < _fMinReliability)
	{
		Toc(4); //validation
		return;
	}

	// Assign the new score!
	ell._score = (score + rel) * 0.5f;
	//ell._score = score;

	// The tentative detection has been confirmed. Save it!
	ellipses.push_back(ell);

	Toc(4); // Validation
};

// Get the coordinates of the center, given the intersection of the estimated lines. See Fig. [8] in Sect [3.2.3] in the paper.
Point2f CEllipseDetectorYaed::GetCenterCoordinates(EllipseData& data_ij, EllipseData& data_ik)
{
	float xx[7];
	float yy[7];

	xx[0] = data_ij.Cab.x;
	xx[1] = data_ik.Cab.x;
	yy[0] = data_ij.Cab.y;
	yy[1] = data_ik.Cab.y;

	{
		//1-1
		float q2 = data_ij.ta;
		float q4 = data_ik.ta;
		Point2f& M12 = data_ij.Ma;
		Point2f& M34 = data_ik.Ma;

		float invDen = 1 / (q2 - q4);
		xx[2] = (M34.y - q4*M34.x - M12.y + q2*M12.x) * invDen;
		yy[2] = (q2*M34.y - q4*M12.y + q2*q4*(M12.x - M34.x)) * invDen;
	}

	{
		//1-2
		float q2 = data_ij.ta;
		float q4 = data_ik.tb;
		Point2f& M12 = data_ij.Ma;
		Point2f& M34 = data_ik.Mb;

		float invDen = 1 / (q2 - q4);
		xx[3] = (M34.y - q4*M34.x - M12.y + q2*M12.x) * invDen;
		yy[3] = (q2*M34.y - q4*M12.y + q2*q4*(M12.x - M34.x)) * invDen;
	}

	{
		//2-2
		float q2 = data_ij.tb;
		float q4 = data_ik.tb;
		Point2f& M12 = data_ij.Mb;
		Point2f& M34 = data_ik.Mb;

		float invDen = 1 / (q2 - q4);
		xx[4] = (M34.y - q4*M34.x - M12.y + q2*M12.x) * invDen;
		yy[4] = (q2*M34.y - q4*M12.y + q2*q4*(M12.x - M34.x)) * invDen;
	}

	{
		//2-1
		float q2 = data_ij.tb;
		float q4 = data_ik.ta;
		Point2f& M12 = data_ij.Mb;
		Point2f& M34 = data_ik.Ma;

		float invDen = 1 / (q2 - q4);
		xx[5] = (M34.y - q4*M34.x - M12.y + q2*M12.x) * invDen;
		yy[5] = (q2*M34.y - q4*M12.y + q2*q4*(M12.x - M34.x)) * invDen;
	}

	xx[6] = (xx[0] + xx[1]) * 0.5f;
	yy[6] = (yy[0] + yy[1]) * 0.5f;


	// Median
	nth_element(xx, xx + 3, xx + 7);
	nth_element(yy, yy + 3, yy + 7);
	float xc = xx[3];
	float yc = yy[3];

	return Point2f(xc, yc);
};


// Verify triplets of arcs with convexity: i=1, j=2, k=4
void CEllipseDetectorYaed::Triplets124(VVP& pi,
	VVP& pj,
	VVP& pk,
	unordered_map<uint, EllipseData>& data,
	vector<Ellipse>& ellipses
	)
{
	// get arcs length
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());
	ushort sz_k = ushort(pk.size());

	// For each edge i
	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		Point& pif = edge_i[0];
		Point& pil = edge_i[sz_ei - 1];

		// 1,2 -> reverse 1, swap
		VP rev_i(edge_i.size());
		reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin());

		// For each edge j
		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			Point& pjf = edge_j[0];
			Point& pjl = edge_j[sz_ej - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
			// CONSTRAINTS on position
			if (pjl.x > pif.x + _fThPosition) //is right	
			{
				//discard
				continue;
			}
#endif

			uint key_ij = GenerateKey(PAIR_12, i, j);

			//for each edge k
			for (ushort k = 0; k < sz_k; ++k)
			{
				VP& edge_k = pk[k];
				ushort sz_ek = ushort(edge_k.size());

				Point& pkf = edge_k[0];
				Point& pkl = edge_k[sz_ek - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
				//CONSTRAINTS on position
				if (pkl.y < pil.y - _fThPosition)
				{
					//discard
					continue;
				}
#endif

				uint key_ik = GenerateKey(PAIR_14, i, k);

				// Find centers

				EllipseData data_ij, data_ik;

				// If the data for the pair i-j have not been computed yet
				if (data.count(key_ij) == 0)
				{
					//1,2 -> reverse 1, swap

					// Compute data!
					GetFastCenter(edge_j, rev_i, data_ij);
					// Insert computed data in the hash table
					data.insert(pair<uint, EllipseData>(key_ij, data_ij));
				}
				else
				{
					// Otherwise, just lookup the data in the hash table
					data_ij = data.at(key_ij);
				}

				// If the data for the pair i-k have not been computed yet
				if (data.count(key_ik) == 0)
				{
					//1,4 -> ok

					// Compute data!
					GetFastCenter(edge_i, edge_k, data_ik);
					// Insert computed data in the hash table
					data.insert(pair<uint, EllipseData>(key_ik, data_ik));
				}
				else
				{
					// Otherwise, just lookup the data in the hash table
					data_ik = data.at(key_ik);
				}

				// INVALID CENTERS
				if (!data_ij.isValid || !data_ik.isValid)
				{
					continue;
				}

#ifndef DISCARD_CONSTRAINT_CENTER
				// Selection strategy - Step 3. See Sect [3.2.2] in the paper
				// The computed centers are not close enough
				if (ed2(data_ij.Cab, data_ik.Cab) > _fMaxCenterDistance2)
				{
					//discard
					continue;
				}
#endif
				// If all constraints of the selection strategy have been satisfied, 
				// we can start estimating the ellipse parameters

				// Find ellipse parameters

				// Get the coordinates of the center (xc, yc)
				Point2f center = GetCenterCoordinates(data_ij, data_ik);

				// Find remaining paramters (A,B,rho)
				FindEllipses(center, edge_i, edge_j, edge_k, data_ij, data_ik, ellipses);
			}
		}
	}
};



void CEllipseDetectorYaed::Triplets231(VVP& pi,
	VVP& pj,
	VVP& pk,
	unordered_map<uint, EllipseData>& data,
	vector<Ellipse>& ellipses
	)
{
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());
	ushort sz_k = ushort(pk.size());

	// For each edge i
	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		Point& pif = edge_i[0];
		Point& pil = edge_i[sz_ei - 1];

		VP rev_i(edge_i.size());
		reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin());

		// For each edge j
		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			Point& pjf = edge_j[0];
			Point& pjl = edge_j[sz_ej - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
			// CONSTRAINTS on position
			if (pjf.y < pif.y - _fThPosition)
			{
				//discard
				continue;
			}
#endif

			VP rev_j(edge_j.size());
			reverse_copy(edge_j.begin(), edge_j.end(), rev_j.begin());

			uint key_ij = GenerateKey(PAIR_23, i, j);

			// For each edge k
			for (ushort k = 0; k < sz_k; ++k)
			{
				VP& edge_k = pk[k];
				ushort sz_ek = ushort(edge_k.size());

				Point& pkf = edge_k[0];
				Point& pkl = edge_k[sz_ek - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
				// CONSTRAINTS on position
				if (pkf.x < pil.x - _fThPosition)
				{
					//discard
					continue;
				}
#endif
				uint key_ik = GenerateKey(PAIR_12, k, i);

				// Find centers

				EllipseData data_ij, data_ik;

				if (data.count(key_ij) == 0)
				{
					// 2,3 -> reverse 2,3

					GetFastCenter(rev_i, rev_j, data_ij);
					data.insert(pair<uint, EllipseData>(key_ij, data_ij));
				}
				else
				{
					data_ij = data.at(key_ij);
				}

				if (data.count(key_ik) == 0)
				{
					// 2,1 -> reverse 1
					VP rev_k(edge_k.size());
					reverse_copy(edge_k.begin(), edge_k.end(), rev_k.begin());

					GetFastCenter(edge_i, rev_k, data_ik);
					data.insert(pair<uint, EllipseData>(key_ik, data_ik));
				}
				else
				{
					data_ik = data.at(key_ik);
				}

				// INVALID CENTERS
				if (!data_ij.isValid || !data_ik.isValid)
				{
					continue;
				}

#ifndef DISCARD_CONSTRAINT_CENTER
				// CONSTRAINT ON CENTERS
				if (ed2(data_ij.Cab, data_ik.Cab) > _fMaxCenterDistance2)
				{
					//discard
					continue;
				}
#endif
				// Find ellipse parameters
				Point2f center = GetCenterCoordinates(data_ij, data_ik);

				FindEllipses(center, edge_i, edge_j, edge_k, data_ij, data_ik, ellipses);

			}
		}
	}
};


void CEllipseDetectorYaed::Triplets342(VVP& pi,
	VVP& pj,
	VVP& pk,
	unordered_map<uint, EllipseData>& data,
	vector<Ellipse>& ellipses
	)
{
	ushort sz_i = ushort(pi.size());
	ushort sz_j = ushort(pj.size());
	ushort sz_k = ushort(pk.size());

	// For each edge i
	for (ushort i = 0; i < sz_i; ++i)
	{
		VP& edge_i = pi[i];
		ushort sz_ei = ushort(edge_i.size());

		Point& pif = edge_i[0];
		Point& pil = edge_i[sz_ei - 1];

		VP rev_i(edge_i.size());
		reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin());

		// For each edge j
		for (ushort j = 0; j < sz_j; ++j)
		{
			VP& edge_j = pj[j];
			ushort sz_ej = ushort(edge_j.size());

			Point& pjf = edge_j[0];
			Point& pjl = edge_j[sz_ej - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
			//CONSTRAINTS on position
			if (pjf.x < pil.x - _fThPosition) 		//is left
			{
				//discard
				continue;
			}
#endif

			VP rev_j(edge_j.size());
			reverse_copy(edge_j.begin(), edge_j.end(), rev_j.begin());

			uint key_ij = GenerateKey(PAIR_34, i, j);

			// For each edge k
			for (ushort k = 0; k < sz_k; ++k)
			{
				VP& edge_k = pk[k];
				ushort sz_ek = ushort(edge_k.size());

				Point& pkf = edge_k[0];
				Point& pkl = edge_k[sz_ek - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
				//CONSTRAINTS on position
				if (pkf.y > pif.y + _fThPosition)
				{
					//discard
					continue;
				}
#endif
				uint key_ik = GenerateKey(PAIR_23, k, i);

				// Find centers

				EllipseData data_ij, data_ik;

				if (data.count(key_ij) == 0)
				{
					//3,4 -> reverse 4

					GetFastCenter(edge_i, rev_j, data_ij);
					data.insert(pair<uint, EllipseData>(key_ij, data_ij));
				}
				else
				{
					data_ij = data.at(key_ij);
				}

				if (data.count(key_ik) == 0)
				{
					//3,2 -> reverse 3,2

					VP rev_k(edge_k.size());
					reverse_copy(edge_k.begin(), edge_k.end(), rev_k.begin());

					GetFastCenter(rev_i, rev_k, data_ik);

					data.insert(pair<uint, EllipseData>(key_ik, data_ik));
				}
				else
				{
					data_ik = data.at(key_ik);
				}


				// INVALID CENTERS
				if (!data_ij.isValid || !data_ik.isValid)
				{
					continue;
				}

#ifndef DISCARD_CONSTRAINT_CENTER
				// CONSTRAINT ON CENTERS
				if (ed2(data_ij.Cab, data_ik.Cab) > _fMaxCenterDistance2)
				{
					//discard
					continue;
				}
#endif
				// Find ellipse parameters
				Point2f center = GetCenterCoordinates(data_ij, data_ik);
				FindEllipses(center, edge_i, edge_j, edge_k, data_ij, data_ik, ellipses);
			}
		}

	}
};



void CEllipseDetectorYaed::Triplets413(VVP& pi,
	VVP& pj,
	VVP& pk,
	unordered_map<uint, EllipseData>& data,
	vector<Ellipse>& ellipses
	)
{
		ushort sz_i = ushort(pi.size());
		ushort sz_j = ushort(pj.size());
		ushort sz_k = ushort(pk.size());

		// For each edge i
		for (ushort i = 0; i < sz_i; ++i)
		{
			VP& edge_i = pi[i];
			ushort sz_ei = ushort(edge_i.size());

			Point& pif = edge_i[0];
			Point& pil = edge_i[sz_ei - 1];

			VP rev_i(edge_i.size());
			reverse_copy(edge_i.begin(), edge_i.end(), rev_i.begin());

			// For each edge j
			for (ushort j = 0; j < sz_j; ++j)
			{
				VP& edge_j = pj[j];
				ushort sz_ej = ushort(edge_j.size());

				Point& pjf = edge_j[0];
				Point& pjl = edge_j[sz_ej - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
				//CONSTRAINTS on position
				if (pjl.y > pil.y + _fThPosition)  		//is below
				{
					//discard
					continue;
				}
#endif

				uint key_ij = GenerateKey(PAIR_14, j, i);

				// For each edge k
				for (ushort k = 0; k < sz_k; ++k)
				{
					VP& edge_k = pk[k];
					ushort sz_ek = ushort(edge_k.size());

					Point& pkf = edge_k[0];
					Point& pkl = edge_k[sz_ek - 1];

#ifndef DISCARD_CONSTRAINT_POSITION
					//CONSTRAINTS on position
					if (pkl.x > pif.x + _fThPosition)
					{
						//discard
						continue;
					}
#endif
					uint key_ik = GenerateKey(PAIR_34, k, i);

					// Find centers

					EllipseData data_ij, data_ik;

					if (data.count(key_ij) == 0)
					{
						// 4,1 -> OK
						GetFastCenter(edge_i, edge_j, data_ij);
						data.insert(pair<uint, EllipseData>(key_ij, data_ij));
					}
					else
					{
						data_ij = data.at(key_ij);
					}

					if (data.count(key_ik) == 0)
					{
						// 4,3 -> reverse 4
						GetFastCenter(rev_i, edge_k, data_ik);
						data.insert(pair<uint, EllipseData>(key_ik, data_ik));
					}
					else
					{
						data_ik = data.at(key_ik);
					}

					// INVALID CENTERS
					if (!data_ij.isValid || !data_ik.isValid)
					{
						continue;
					}

#ifndef DISCARD_CONSTRAINT_CENTER
					// CONSTRAIN ON CENTERS
					if (ed2(data_ij.Cab, data_ik.Cab) > _fMaxCenterDistance2)
					{
						//discard
						continue;
					}
#endif
					// Find ellipse parameters
					Point2f center = GetCenterCoordinates(data_ij, data_ik);

					FindEllipses(center, edge_i, edge_j, edge_k, data_ij, data_ik, ellipses);

				}
			}
		}
	};


void CEllipseDetectorYaed::RemoveShortEdges(Mat1b& edges, Mat1b& clean)
{
	VVP contours;

	// Labeling and contraints on length
	Labeling(edges, contours, _iMinEdgeLength);

	int iContoursSize = contours.size();
	for (int i = 0; i < iContoursSize; ++i)
	{
		VP& edge = contours[i];
		unsigned szEdge = edge.size();

		// Constraint on axes aspect ratio
		RotatedRect oriented = minAreaRect(edge);
		if (oriented.size.width < _fMinOrientedRectSide ||
			oriented.size.height < _fMinOrientedRectSide ||
			oriented.size.width > oriented.size.height * _fMaxRectAxesRatio ||
			oriented.size.height > oriented.size.width * _fMaxRectAxesRatio)
		{
			continue;
		}

		for (unsigned j = 0; j < szEdge; ++j)
		{
			clean(edge[j]) = (uchar)255;
		}
	}
}



void CEllipseDetectorYaed::PrePeocessing(Mat1b& I,
	Mat1b& DP,
	Mat1b& DN
	)
{

	Tic(0); //edge detection

	// Smooth image
	GaussianBlur(I, I, _szPreProcessingGaussKernelSize, _dPreProcessingGaussSigma);

	// Temp variables
	Mat1b E;				//edge mask
	Mat1s DX, DY;			//sobel derivatives

	// Detect edges
	Canny3(I, E, DX, DY, 3, false);

	Toc(0); //edge detection

	Tac(1); //preprocessing

	// For each edge points, compute the edge direction
	for (int i = 0; i<_szImg.height; ++i)
	{
		short* _dx = DX.ptr<short>(i);
		short* _dy = DY.ptr<short>(i);
		uchar* _e = E.ptr<uchar>(i);
		uchar* _dp = DP.ptr<uchar>(i);
		uchar* _dn = DN.ptr<uchar>(i);

		for (int j = 0; j<_szImg.width; ++j)
		{
			if (!((_e[j] <= 0) || (_dx[j] == 0) || (_dy[j] == 0)))
			{
				// Angle of the tangent
				float phi = -(float(_dx[j]) / float(_dy[j]));

				// Along positive or negative diagonal
				if (phi > 0)	_dp[j] = (uchar)255;
				else if (phi < 0)	_dn[j] = (uchar)255;
			}
		}
	}
};


void CEllipseDetectorYaed::DetectAfterPreProcessing(vector<Ellipse>& ellipses, Mat1b& E, const Mat1f& PHI)
{
	// Set the image size
	_szImg = E.size();

	// Initialize temporary data structures
	Mat1b DP = Mat1b::zeros(_szImg);		// arcs along positive diagonal
	Mat1b DN = Mat1b::zeros(_szImg);		// arcs along negative diagonal

	// For each edge points, compute the edge direction
	for (int i = 0; i<_szImg.height; ++i)
	{
		const float* _phi = PHI.ptr<float>(i);
		uchar* _e = E.ptr<uchar>(i);
		uchar* _dp = DP.ptr<uchar>(i);
		uchar* _dn = DN.ptr<uchar>(i);

		for (int j = 0; j<_szImg.width; ++j)
		{
			if ((_e[j] > 0) && (_phi[j] != 0))
			{
				// Angle
				
				// along positive or negative diagonal
				if (_phi[j] > 0)	_dp[j] = (uchar)255;
				else if (_phi[j] < 0)	_dn[j] = (uchar)255;
			}
		}
	}

	// Initialize accumulator dimensions
	ACC_N_SIZE = 101;
	ACC_R_SIZE = 180;
	ACC_A_SIZE = max(_szImg.height, _szImg.width);

	// Allocate accumulators
	accN = new int[ACC_N_SIZE];
	accR = new int[ACC_R_SIZE];
	accA = new int[ACC_A_SIZE];

	// Other temporary 
	VVP points_1, points_2, points_3, points_4;		//vector of points, one for each convexity class
	unordered_map<uint, EllipseData> centers;		//hash map for reusing already computed EllipseData

	// Detect edges and find convexities
	DetectEdges13(DP, points_1, points_3);
	DetectEdges24(DN, points_2, points_4);

	// Find triplets
	Triplets124(points_1, points_2, points_4, centers, ellipses);
	Triplets231(points_2, points_3, points_1, centers, ellipses);
	Triplets342(points_3, points_4, points_2, centers, ellipses);
	Triplets413(points_4, points_1, points_3, centers, ellipses);

	// Sort detected ellipses with respect to score
	sort(ellipses.begin(), ellipses.end());

	//free accumulator memory
	delete[] accN;
	delete[] accR;
	delete[] accA;

	//cluster detections
	//ClusterEllipses(ellipses);
};


void CEllipseDetectorYaed::Detect(Mat1b& I, vector<Ellipse>& ellipses)
{
	Tic(1); //prepare data structure

	// Set the image size
	_szImg = I.size();

	// Initialize temporary data structures
	Mat1b DP = Mat1b::zeros(_szImg);		// arcs along positive diagonal
	Mat1b DN = Mat1b::zeros(_szImg);		// arcs along negative diagonal

	// Initialize accumulator dimensions
	ACC_N_SIZE = 101;
	ACC_R_SIZE = 180;
	ACC_A_SIZE = max(_szImg.height, _szImg.width);

	// Allocate accumulators
	accN = new int[ACC_N_SIZE];
	accR = new int[ACC_R_SIZE];
	accA = new int[ACC_A_SIZE];

	// Other temporary 
	VVP points_1, points_2, points_3, points_4;		//vector of points, one for each convexity class
	unordered_map<uint, EllipseData> centers;		//hash map for reusing already computed EllipseData

	Toc(1); //prepare data structure

	// Preprocessing
	// From input image I, find edge point with coarse convexity along positive (DP) or negative (DN) diagonal
	PrePeocessing(I, DP, DN);

	// Detect edges and find convexities
	DetectEdges13(DP, points_1, points_3);
	DetectEdges24(DN, points_2, points_4);

	Toc(1); //preprocessing


	// DEBUG
	Mat3b out(I.rows, I.cols, Vec3b(0,0,0));
	for(unsigned i=0; i<points_1.size(); ++i)
	{
		//Vec3b color(rand()%255, 128+rand()%127, 128+rand()%127);
		Vec3b color(255,0,0);
		for(unsigned j=0; j<points_1[i].size(); ++j)
			out(points_1[i][j]) = color;
	}

	for(unsigned i=0; i<points_2.size(); ++i)
	{
		//Vec3b color(rand()%255, 128+rand()%127, 128+rand()%127);
		Vec3b color(0,255,0);
		for(unsigned j=0; j<points_2[i].size(); ++j)
			out(points_2[i][j]) = color;
	}
	for(unsigned i=0; i<points_3.size(); ++i)
	{
		//Vec3b color(rand()%255, 128+rand()%127, 128+rand()%127);
		Vec3b color(0,0,255);
		for(unsigned j=0; j<points_3[i].size(); ++j)
			out(points_3[i][j]) = color;
	}

	for(unsigned i=0; i<points_4.size(); ++i)
	{
		//Vec3b color(rand()%255, 128+rand()%127, 128+rand()%127);
		Vec3b color(255,0,255);
		for(unsigned j=0; j<points_4[i].size(); ++j)
			out(points_4[i][j]) = color;
	}


	// time estimation, validation  inside

	Tic(2); //grouping
	//find triplets
	Triplets124(points_1, points_2, points_4, centers, ellipses);
	Triplets231(points_2, points_3, points_1, centers, ellipses);
	Triplets342(points_3, points_4, points_2, centers, ellipses);
	Triplets413(points_4, points_1, points_3, centers, ellipses);
	Toc(2); //grouping	
	// time estimation, validation inside
	_times[2] -= (_times[3] + _times[4]);

	Tac(4); //validation
	// Sort detected ellipses with respect to score
	sort(ellipses.begin(), ellipses.end());
	Toc(4); //validation


	// Free accumulator memory
	delete[] accN;
	delete[] accR;
	delete[] accA;

	Tic(5);
	// Cluster detections
	ClusterEllipses(ellipses);
	Toc(5);

};




// Ellipse clustering procedure. See Sect [3.3.2] in the paper.
void CEllipseDetectorYaed::ClusterEllipses(vector<Ellipse>& ellipses)
{
	float th_Da = 0.1f;
	float th_Db = 0.1f;
	float th_Dr = 0.1f;

	float th_Dc_ratio = 0.1f;
	float th_Dr_circle = 0.9f;

	int iNumOfEllipses = int(ellipses.size());
	if (iNumOfEllipses == 0) return;

	// The first ellipse is assigned to a cluster
	vector<Ellipse> clusters;
	clusters.push_back(ellipses[0]);

	bool bFoundCluster = false;

	for (int i = 1; i<iNumOfEllipses; ++i)
	{
		Ellipse& e1 = ellipses[i];

		int sz_clusters = int(clusters.size());

		float ba_e1 = e1._b / e1._a;
		float Decc1 = e1._b / e1._a;

		bool bFoundCluster = false;
		for (int j = 0; j<sz_clusters; ++j)
		{
			Ellipse& e2 = clusters[j];

			float ba_e2 = e2._b / e2._a;
			float th_Dc = min(e1._b, e2._b) * th_Dc_ratio;
			th_Dc *= th_Dc;

			// Centers
			float Dc = ((e1._xc - e2._xc)*(e1._xc - e2._xc) + (e1._yc - e2._yc)*(e1._yc - e2._yc));
			if (Dc > th_Dc)
			{
				//not same cluster
				continue;
			}

			// a
			float Da = abs(e1._a - e2._a) / max(e1._a, e2._a);
			if (Da > th_Da)
			{
				//not same cluster
				continue;
			}

			// b
			float Db = abs(e1._b - e2._b) / min(e1._b, e2._b);
			if (Db > th_Db)
			{
				//not same cluster
				continue;
			}

			// angle
			float Dr = GetMinAnglePI(e1._rad, e2._rad) / float(CV_PI);
			if ((Dr > th_Dr) && (ba_e1 < th_Dr_circle) && (ba_e2 < th_Dr_circle))
			{
				//not same cluster
				continue;
			}

			// Same cluster as e2
			bFoundCluster = true;
			// Discard, no need to create a new cluster
			break;
		}

		if (!bFoundCluster)
		{
			// Create a new cluster			
			clusters.push_back(e1);
		}
	}

	clusters.swap(ellipses);
};



//Draw at most iTopN detected ellipses.
void CEllipseDetectorYaed::DrawDetectedEllipses(Mat3b& output, vector<coordinate>& ellipse_out, vector<Ellipse>& ellipses, int iTopN, int thickness) {

/*
	int sz_ell = int(ellipses.size());
	int n = (iTopN == 0) ? sz_ell : min(iTopN, sz_ell);
    cout<<"the number of ellipse:"<<n<<endl;
	for (int i = 0; i < n; ++i)
	{
		Ellipse& e = ellipses[n - i - 1];
		int g = cvRound(e._score * 255.f);
		Scalar color(0, g, 0);
		ellipse(output, Point(cvRound(e._xc), cvRound(e._yc)), Size(cvRound(e._a), cvRound(e._b)), e._rad*180.0 / CV_PI, 0.0, 360.0, color, thickness);
        cout<<"the coordinate of the ellipse:"<<endl<<"x:"<<e._xc<<endl<<"y:"<<e._yc<<endl<<"a:"<<e._a<<endl<<"scores:"<<e._score<<endl;
	}
	 */

	//用冒泡法将得到的椭圆序列按照x方向坐标由小到大排序

	int n_e = ellipses.size();
//	for (int i = 0; i < n_e - 1; i++) {
//		for (int j = 0; j < n_e - 1 - i; j++) {
//			if (ellipses[j]._xc > ellipses[j + 1]._xc) {
//				swap(ellipses[j], ellipses[j + 1]);
//			}
//		}
//	}
	//将容器中的椭圆按照同心圆与否排列
//	vector<Ellipse> v;
//	int f = 0;
//	for (auto i = 0; i < n_e;f == 0?i = i+1:i = i+2) {
//		f = 0;
//		if (abs(ellipses[i]._xc -ellipses [i + 1]._xc) < 20 && abs(ellipses[i]._yc - ellipses[i + 1]._yc) < 20) {
//			if (ellipses[i]._a < ellipses[i + 1]._a) {
//				v.push_back(ellipses[i + 1]);
//				v.push_back(ellipses[i]);
//				f = 1;
//			} else {
//				v.push_back(ellipses[i]);
//				v.push_back(ellipses[i + 1]);
//				f = 1;
//			}
//		} else {
//			v.push_back(ellipses[i]);
//			v.push_back(Ellipse());
//		}
//	}
	//绘制椭圆
	for(auto i = 0; i < ellipses.size(); i = i +1) {
		Scalar color(0, 255, 0);
		Ellipse& e = ellipses[i];
		ellipse(output, Point(cvRound(e._xc), cvRound(e._yc)), Size(cvRound(e._a), cvRound(e._b)),
				e._rad * 180.0 / CV_PI, 0.0, 360.0, color, thickness);
		const string text = to_string(i);
		putText(output, text, Point(cvRound(e._xc), cvRound(e._yc)),CV_FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),2,8);//给目标编号

        coordinate e_c;
        e_c.x = e._xc;
        e_c.y = e._yc;
        e_c.order = i;
        e_c.a = e._a;
        ellipse_out.push_back(e_c);

	}
}

void CEllipseDetectorYaed::OptimizEllipse(vector<Ellipse> &ellipse_out, vector<Ellipse> &ellipses_in){
	float score = 0.74;
	/***************************去掉评分不佳的椭圆********************************************/
	vector<Ellipse> e0;//存放去掉评分小于0.8的椭圆
	for(auto i = ellipses_in.begin(); i != ellipses_in.end(); ++i){
		if((*i)._score < score)
			continue;
		else
			e0.push_back(*i);
	}

	/*************延x轴方向对椭圆由小到大排序**************************/
//	for (auto i = 0; i < (e0.size() - 1); i++) {
//		for(auto j = 0; j < (e0.size() - 1 - i); j++){
//			if (e0[j]._xc > e0[j + 1]._xc) {
//				swap(e0[j], e0[j + 1]);
//			}
//		}
//	}
/*感觉这个程序和上面的没啥区别，不知道为啥就是通不过*/
	int n_e = e0.size();
	for (int i = 0; i < n_e - 1; i++) {
		for (int j = 0; j < n_e - 1 - i; j++) {
			if (e0[j]._xc > e0[j + 1]._xc) {
				swap(e0[j], e0[j + 1]);
			}
		}
	}
	/*************判断同心圆，只留下大圆**************************/
	vector<Ellipse> v, v1;
	int f = 0;
	for (auto i = 0; i < n_e;f == 0? i = i+1:i = i+2) {
		f = 0;
		if (abs(e0[i]._xc - e0[i + 1]._xc) < 20 && abs(e0[i]._yc - e0[i + 1]._yc) < 20) {
			if (e0[i]._a < e0[i + 1]._a) {
				v.push_back(e0[i + 1]);
				f = 1;
			} else {
				v.push_back(e0[i]);
				f = 1;
			}
		} else {
			v.push_back(e0[i]);
		}
	}
	for(auto &p:v){
		if(v1.size() == 0){
			v1.push_back(p);
		} else{
			for(auto j = 0; j <v1.size(); j++ ){
				if (abs(v1[j]._xc - p._xc) < 20 && abs(v1[j]._yc - p._yc) < 20)
					break;
				else if( j != ( v1.size() - 1)){
					continue;
				} else{
					v1.push_back(p);
					break;
				}
			}
		}
	}
	/***********************按照左下、左上、右下、右上的顺序对椭圆排序******************************/
	vector<Ellipse> left,right;
	for (auto &p:v1) {
		if(p._xc < 320)
			left.push_back(p);
		else
			right.push_back(p);
	}
	int l = left.size();
	int r = right.size();
	for (int i = 0; i < l - 1; i++) {
		for (int j = 0; j < l - 1 - i; j++) {
			if (left[j]._yc < left[j + 1]._yc) {
				swap(left[j], left[j + 1]);
			}
		}
	}
	for (int i = 0; i < r - 1; i++) {
		for (int j = 0; j < r - 1 - i; j++) {
			if (right[j]._yc < right[j + 1]._yc) {
				swap(right[j], right[j + 1]);
			}
		}
	}
	ellipse_out = left;
	for(auto &p:right){
		ellipse_out.push_back(p);
	}

}

void CEllipseDetectorYaed::extracrROI(Mat1b& image, vector<coordinate>& ellipse_out, vector<Mat1b>& img_roi){

//	cvtColor(image, image, COLOR_RGB2GRAY);
	GaussianBlur(image, image, Size(5, 5),0, 0);
	threshold(image, image, 140, 255, CV_THRESH_BINARY);
	for(auto &p:ellipse_out){
		Mat1b ROI;
		int r = 0.6 * p.a;
		int x_l = 3 * p.x - r;
		int y_l = 3 * p.y - r;
		int width = 2 * r;
		ROI = image(Rect(x_l, y_l, width, width));
		cout<<"ROI.size = "<<ROI.size<<endl;
		namedWindow("ROI",1);
		imshow("ROI", ROI);
		img_roi.push_back(ROI);
	}
}

void visual_rec(vector<Mat1b>& gray, vector<coordinate>& ellipse_out0, vector<coordinate>& ellipse_out00, vector< vector<Point> >& contours0){
    float areanum = 0.215;
//    threshold(gray, gray, 120, 255, CV_THRESH_BINARY);
//  imshow("threshold", thresh);
//  morphologyEx(gauss, gauss, MORPH_CLOSE, (5, 5) );
//    Canny(thresh, canny, 50, 150, 3);
		vector< vector<Point> > contours;

    for(auto j = 0; j < gray.size(); j++) {
		findContours(gray[j], contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    	for (int i = 0; i < contours.size(); i++) {
            //拟合出轮廓外侧最小的矩形
            RotatedRect rotate_rect = minAreaRect(contours[i]);
            Point2f *vertices = new Point2f[4];
            rotate_rect.points(vertices);
            float h1, h2, h3, h4;
            h1 = 0.4 * gray[j].cols;
            h2 = 0.9 * gray[j].cols;
            h3 = 0.314 * gray[j].cols;
            h4 = 0.5 * gray[j].cols;
            if (rotate_rect.size.height < h1 || rotate_rect.size.height > h2 || rotate_rect.size.width < h1 || rotate_rect.size.width > h2
                || abs(rotate_rect.center.x - h4) > h3 || abs(rotate_rect.center.y - h4) > h3)
				continue;

            	float x12 = (vertices[1].x + vertices[2].x) / 2;
				float y12 = (vertices[1].y + vertices[2].y) / 2;
				float xt12 = areanum * (rotate_rect.center.x - x12) + x12;
				float yt12 = y12 - areanum * (y12 - rotate_rect.center.y);

				float x30 = (vertices[3].x + vertices[0].x) / 2;
				float y30 = (vertices[3].y + vertices[0].y) / 2;
				float yt30 = areanum * (rotate_rect.center.y - y30) + y30;
				float xt30 = x30 - areanum * (x30 - rotate_rect.center.x);

				float x23 = (vertices[2].x + vertices[3].x) / 2;
				float y23 = (vertices[2].y + vertices[3].y) / 2;
				float xt23 = areanum * (rotate_rect.center.x - x23) + x23;
				float yt23 = y23 - areanum * (y23 - rotate_rect.center.y);

				float x01 = (vertices[1].x + vertices[0].x) / 2;
				float y01 = (vertices[1].y + vertices[0].y) / 2;
				float yt01 = areanum * (rotate_rect.center.y - y01) + y01;
				float xt01 = x01 - areanum * (x01 - rotate_rect.center.x);

				if (abs((gray[j].at<uchar>(yt12, xt12) - gray[j].at<uchar>(yt30, xt30))) < 60
					&& abs((gray[j].at<uchar>(yt23, xt23) - gray[j].at<uchar>(yt01, xt01))) < 60) {
					ellipse_out0[j].flag = true;
				} else {
					ellipse_out0[j].flag = false;
				}
				vector<Point> contour;
				for (int i = 0; i < 4; i++) {
					vertices[i].x = vertices[i].x + (3 * ellipse_out0[j].x - 0.6 * ellipse_out0[j].a);
					vertices[i].y = vertices[i].y + (3 * ellipse_out0[j].y - 0.6 * ellipse_out0[j].a);
					contour.push_back(vertices[i]);
				}
				contours0.push_back(contour);
        }
        ellipse_out00.push_back(ellipse_out0[j]);
    }
}

/*
void possibility(uint8_t image[] , float poss[])
{
    const int len = 1024;
    float poss_F = 0.0;
    float poss_T = 0.0;
    float scale = 100000.0;

    float weights_1[len] = { -10.2875320000000,-14.1449560000000,-14.3858580000000,-13.0162360000000,-13.4349080000000,-13.3815360000000,-13.2282770000000,-12.7502570000000,-12.5296050000000,-12.9897790000000,-13.3035690000000,-13.4241910000000,-13.1837540000000,-12.8393200000000,-12.2122500000000,-12.1045520000000,-12.0070450000000,-11.8322680000000,-11.3967790000000,-11.3962510000000,-11.8383950000000,-13.0766110000000,-13.8351210000000,-14.2863860000000,-14.1427030000000,-14.0916990000000,-12.7171650000000,-10.2403490000000,-7.58099650000000,-7.02192450000000,-7.79563090000000,-6.98546410000000,-12.7442860000000,-14.0121280000000,-12.8339320000000,-13.1048090000000,-13.3193430000000,-12.7841810000000,-12.4766500000000,-12.1458360000000,-12.5623030000000,-13.0952990000000,-13.2983380000000,-13.3000000000000,-12.8466190000000,-12.6063420000000,-12.3242940000000,-12.1714260000000,-11.8982480000000,-11.7422840000000,-11.7244590000000,-12.0314540000000,-12.6523820000000,-13.3510650000000,-14.2290180000000,-14.6021220000000,-13.9108050000000,-13.2733880000000,-13.0315000000000,-12.3122620000000,-9.91479400000000,-7.61706970000000,-7.69789890000000,-8.01844310000000,-13.1398920000000,-13.4768850000000,-12.7548760000000,-13.2377280000000,-11.9631560000000,-11.1981340000000,-11.3482100000000,-11.9212720000000,-12.2388110000000,-11.3737850000000,-9.87259770000000,-10.1550800000000,-10.2353560000000,-10.2091050000000,-10.9487740000000,-11.1999130000000,-11.4512510000000,-11.5742230000000,-11.8176800000000,-12.2795560000000,-13.1461500000000,-14.0294420000000,-14.5151220000000,-14.0781940000000,-12.9953470000000,-11.9838190000000,-12.1075150000000,-13.2060040000000,-12.2763540000000,-9.61912160000000,-7.66421790000000,-7.98867080000000,-14.3317600000000,-13.2553460000000,-15.0396280000000,-17.0766450000000,-15.0620670000000,-12.7489630000000,-11.4288080000000,-12.4154320000000,-12.1444960000000,-5.29219480000000,-0.199637140000000,0.537828150000000,0.454600130000000,-1.49552980000000,-4.36000000000000,-6.35287480000000,-7.16951230000000,-7.27227500000000,-7.39695070000000,-7.89921050000000,-8.30488970000000,-9.29039290000000,-9.88803100000000,-9.68091870000000,-8.39845750000000,-7.61692760000000,-9.86686330000000,-12.9522030000000,-12.8889010000000,-11.5799520000000,-8.65294740000000,-7.43142370000000,-14.3615810000000,-14.4507410000000,-16.2790450000000,-19.9490620000000,-21.4973200000000,-22.5586340000000,-21.0401100000000,-22.5276450000000,-20.5701640000000,-8.03005120000000,7.37074280000000,14.7907630000000,16.9762880000000,17.4325660000000,13.9514170000000,10.1742200000000,7.74626590000000,7.63008360000000,2.33487840000000,2.00526550000000,2.78624440000000,2.45865700000000,0.409658550000000,-0.541567800000000,1.69522360000000,2.16636590000000,-1.07829570000000,-6.54704570000000,-8.41245840000000,-9.92114640000000,-9.78517720000000,-6.55577950000000,-14.0088540000000,-15.4457780000000,-16.5086780000000,-20.7947790000000,-30.1768380000000,-45.3671380000000,-43.2535250000000,-43.6233830000000,-40.8439830000000,-23.0571350000000,6.44972130000000,23.9647670000000,29.1205500000000,33.4768450000000,35.2541390000000,35.6831090000000,41.9910280000000,39.7716940000000,28.8186840000000,20.7350640000000,18.1599750000000,16.4227560000000,10.5090190000000,5.55831530000000,4.73934080000000,8.27559090000000,4.92078780000000,-1.15063510000000,-3.91827850000000,-7.94642690000000,-10.6006690000000,-7.86095430000000,-14.8697180000000,-15.3832630000000,-16.7364060000000,-20.7492830000000,-35.1952020000000,-51.2638550000000,-43.8757440000000,-41.5058900000000,-37.6547780000000,-19.4292320000000,9.59464550000000,31.0608370000000,36.1622890000000,44.1881560000000,53.0838620000000,62.5313260000000,73.1896060000000,73.4411540000000,69.8049850000000,58.0739210000000,43.8507500000000,32.4945450000000,25.0208020000000,19.4671290000000,16.6135880000000,12.4893030000000,6.27184440000000,-0.285999180000000,-3.52413230000000,-9.29066660000000,-11.9465710000000,-10.6829850000000,-15.3353030000000,-14.7836730000000,-16.7252790000000,-22.1670270000000,-36.8356670000000,-51.9731790000000,-42.3743360000000,-34.0245700000000,-27.6364590000000,-12.3880780000000,14.5223060000000,36.2950400000000,39.6488650000000,45.5871200000000,56.3252980000000,75.9779210000000,84.3983000000000,89.6424790000000,92.8117900000000,86.8518070000000,67.5751340000000,46.1172560000000,36.3110850000000,29.4854640000000,25.6458230000000,18.7073230000000,7.24673560000000,0.213100700000000,-4.16112140000000,-10.9824250000000,-12.7769030000000,-12.2177560000000,-14.4167070000000,-13.9741910000000,-15.9632930000000,-19.6928790000000,-33.3955350000000,-51.2553830000000,-39.1089900000000,-25.6278420000000,-20.4445480000000,-11.5369020000000,16.6773190000000,34.8041880000000,40.4227830000000,50.0336460000000,69.1290050000000,82.5377960000000,86.7203830000000,93.9321370000000,100.497050000000,92.8188780000000,70.1975100000000,58.4707760000000,46.5432660000000,37.4594190000000,29.7930300000000,21.5151960000000,9.81611920000000,3.36485150000000,-3.04853200000000,-11.7856390000000,-13.3049580000000,-13.2238130000000,-13.6341760000000,-13.9242290000000,-15.8057660000000,-18.0543980000000,-27.6246700000000,-48.1816900000000,-33.8474200000000,-18.1568780000000,-19.6578010000000,-13.3942050000000,14.4727000000000,32.4847600000000,36.5871540000000,55.2395210000000,79.8490910000000,87.3815230000000,90.5212100000000,93.8147130000000,95.9879990000000,81.7211000000000,75.5272830000000,64.1961060000000,53.2207340000000,46.0685460000000,40.6559330000000,29.7918950000000,12.9000000000000,6.68859720000000,-2.57076100000000,-11.7581790000000,-13.4318480000000,-14.0001290000000,-13.4705990000000,-13.8595720000000,-14.0839560000000,-16.1691970000000,-24.1184670000000,-44.1231610000000,-23.4118560000000,-9.37799740000000,-15.1743290000000,-12.4036420000000,6.79353190000000,16.2829400000000,13.7558570000000,36.9706500000000,53.6764410000000,71.0666200000000,87.6082080000000,93.1799160000000,84.1588440000000,75.9516300000000,75.4070820000000,58.8119050000000,44.6286890000000,39.7380450000000,45.7597050000000,42.6746640000000,24.5651860000000,12.7909130000000,-2.52128310000000,-10.6944600000000,-13.1117790000000,-14.0150160000000,-12.5629510000000,-12.3589440000000,-12.6591340000000,-15.4295960000000,-21.4345510000000,-34.5070300000000,-14.9066310000000,-6.53580140000000,-10.6069220000000,-5.88996220000000,4.40534210000000,-1.41152980000000,-2.53922510000000,3.78260160000000,9.17152120000000,28.6422560000000,70.0208820000000,85.9682690000000,74.3317950000000,72.7321930000000,68.6053620000000,44.1467060000000,22.8229410000000,22.3261320000000,28.6030460000000,24.8972970000000,23.4171640000000,23.0651610000000,2.29329680000000,-7.63827510000000,-12.3570650000000,-14.0537220000000,-11.8680350000000,-12.0478570000000,-12.2400180000000,-14.5237640000000,-18.0784870000000,-22.7642460000000,-11.1152000000000,-2.19555900000000,-7.06000000000000,-6.53754430000000,-1.50125810000000,-10.1769120000000,-12.1315530000000,-18.9389820000000,-18.7104660000000,-7.98653890000000,31.2183990000000,48.6121860000000,56.4353490000000,58.9088480000000,52.8010180000000,25.7124540000000,7.59605500000000,16.2490180000000,14.6927130000000,10.5418900000000,12.2180850000000,19.8058660000000,5.81894060000000,-5.56025220000000,-11.2862690000000,-13.6942990000000,-11.9877740000000,-12.2773580000000,-12.3187250000000,-13.7616440000000,-15.0445970000000,-11.5669840000000,-3.45073490000000,-0.121756510000000,-7.06211190000000,-15.1745200000000,-23.0416550000000,-41.1765980000000,-54.1736220000000,-58.3753700000000,-59.3133930000000,-48.9118040000000,-18.5112950000000,-4.36144590000000,16.8541680000000,24.4399800000000,23.5088100000000,2.70115350000000,-6.89808320000000,-3.09682700000000,-5.93113180000000,-9.29116730000000,-5.59664540000000,-0.529506740000000,-6.23647210000000,-8.83392140000000,-11.8537000000000,-13.6524360000000,-11.8463830000000,-12.5497970000000,-12.3565240000000,-11.9111100000000,-9.02993970000000,2.16508650000000,16.0084360000000,8.37528130000000,-7.85297870000000,-13.7414400000000,-27.5033360000000,-62.2525560000000,-80.3350370000000,-87.8876950000000,-91.0802610000000,-84.5244600000000,-56.8059390000000,-28.3479370000000,-15.0898220000000,-6.24889420000000,0.200336220000000,-6.03038260000000,-24.4424800000000,-25.5837020000000,-23.3489360000000,-21.5289860000000,-20.0857330000000,-17.1365850000000,-9.57213310000000,-11.1579020000000,-13.1163980000000,-13.3032760000000,-11.7337960000000,-12.5948330000000,-12.2596290000000,-9.01800160000000,3.64464550000000,11.9563830000000,21.7814560000000,18.3289740000000,4.64061500000000,-6.45890430000000,-22.3380410000000,-69.8645250000000,-90.4579010000000,-98.2261350000000,-106.176090000000,-113.071080000000,-68.7807920000000,-35.7985420000000,-26.8405270000000,-17.8975220000000,-4.81558280000000,-9.49648950000000,-42.4444310000000,-45.5399970000000,-41.3422780000000,-35.2440950000000,-39.4489060000000,-27.6249050000000,-13.1130650000000,-11.9098110000000,-13.5071260000000,-13.1275820000000,-12.0725570000000,-12.1137920000000,-10.7315780000000,-2.66554520000000,16.3143220000000,17.8429240000000,22.0247360000000,18.6266380000000,12.2646060000000,2.74243380000000,-25.6165560000000,-76.2785570000000,-92.4256290000000,-96.1424940000000,-111.186430000000,-118.766500000000,-64.5479580000000,-39.0248150000000,-27.1705000000000,-16.4555470000000,7.95510670000000,-10.3572890000000,-47.3376160000000,-49.8915860000000,-44.7148060000000,-47.3849530000000,-54.5352550000000,-35.0190850000000,-18.0382290000000,-14.6848260000000,-13.1287520000000,-12.5090100000000,-11.9839730000000,-10.9967670000000,-8.69253350000000,2.74912830000000,20.8183480000000,19.5694680000000,21.0318660000000,18.7745420000000,10.9038620000000,-1.15524230000000,-33.6212730000000,-74.3076930000000,-85.8994600000000,-95.1991350000000,-112.108600000000,-100.236690000000,-49.7788160000000,-35.1814270000000,-26.1545200000000,-4.39804940000000,16.3137510000000,-13.9738000000000,-48.0117110000000,-46.8747410000000,-42.6366620000000,-59.2439120000000,-59.5568280000000,-39.6064640000000,-19.9469890000000,-15.4159310000000,-13.3303870000000,-11.7805920000000,-10.9804340000000,-9.48876380000000,-6.30797620000000,0.974374770000000,11.8977130000000,16.3112580000000,20.0269110000000,16.9509010000000,11.2174860000000,-10.7833220000000,-32.6033250000000,-57.6312870000000,-64.9821850000000,-77.8209000000000,-91.2962800000000,-66.2843020000000,-26.3205800000000,-15.3055990000000,-2.72630480000000,17.9165880000000,14.4954090000000,-19.9655690000000,-47.6542280000000,-42.3962140000000,-47.8420450000000,-60.3783610000000,-55.4700930000000,-41.2309650000000,-22.6327340000000,-15.5474900000000,-13.5000000000000,-11.2386960000000,-10.8603530000000,-8.24578760000000,-3.68534040000000,2.77086380000000,6.18794680000000,0.739828170000000,9.85671810000000,13.8427170000000,8.14455510000000,-10.0143730000000,-10.7701160000000,-15.9264240000000,-18.6083640000000,-24.6162930000000,-25.0743260000000,-8.54823970000000,13.2300940000000,20.5728190000000,41.9807360000000,51.7869000000000,32.3235130000000,-7.07040740000000,-32.1947140000000,-32.7267040000000,-45.1365200000000,-48.2108760000000,-43.7780530000000,-34.5910840000000,-21.6283890000000,-15.8153070000000,-13.3847940000000,-10.6925850000000,-12.0201600000000,-7.79224400000000,-0.682956580000000,3.65178700000000,4.20344500000000,-6.56116960000000,-2.54485010000000,-2.30214310000000,-3.75791570000000,-6.72981500000000,0.898022650000000,1.92014740000000,3.24920850000000,5.35275790000000,21.3243660000000,43.7075420000000,57.0141220000000,59.2200320000000,74.8918530000000,72.0092320000000,46.5576020000000,12.1347940000000,-14.8886360000000,-32.9930080000000,-36.8571700000000,-36.2659150000000,-29.4469740000000,-20.5832040000000,-12.9690430000000,-11.2652540000000,-10.0251100000000,-8.59444330000000,-11.7336560000000,-8.00060840000000,-0.393101300000000,3.68012240000000,-0.102618350000000,-13.9264070000000,-10.2779210000000,-18.5531560000000,-28.0253520000000,-24.5484520000000,0.562779780000000,-1.54876150000000,0.885263380000000,16.9221730000000,37.9424510000000,59.2140240000000,79.1847080000000,82.6550980000000,77.7083970000000,61.3438000000000,41.9930570000000,15.0882440000000,-19.9290370000000,-33.5856550000000,-31.7025510000000,-33.8539540000000,-27.9956680000000,-16.6790750000000,-11.1663540000000,-8.95187760000000,-7.58769370000000,-6.81970410000000,-12.4149870000000,-10.0075540000000,-4.89007090000000,0.159252780000000,-4.71605870000000,-21.3786620000000,-20.7519550000000,-30.2113290000000,-39.2575190000000,-42.1352810000000,-15.4946560000000,-5.93538380000000,7.47196440000000,26.7503450000000,43.3072280000000,63.9223480000000,84.3570940000000,88.9804530000000,75.9371640000000,56.4730610000000,40.1726420000000,7.25076870000000,-27.7296700000000,-30.1665520000000,-27.9423710000000,-29.4259780000000,-24.5400560000000,-15.8852430000000,-11.4269290000000,-8.15151500000000,-6.55093340000000,-5.90693330000000,-12.5978560000000,-12.4499990000000,-9.20405960000000,-5.36375670000000,-11.9187120000000,-28.1966170000000,-27.9916590000000,-36.0319370000000,-42.7421680000000,-40.0785600000000,-18.7044470000000,-17.6878680000000,3.67297510000000,33.6770820000000,48.4512980000000,65.2489470000000,83.2671280000000,86.5216060000000,73.9463650000000,56.4516640000000,37.4474260000000,-2.24098750000000,-27.9905990000000,-28.0019440000000,-27.2776070000000,-30.0395680000000,-24.0129430000000,-15.5349400000000,-11.8140750000000,-9.17043690000000,-6.34278630000000,-4.83467630000000,-12.4002050000000,-12.7133790000000,-9.90634060000000,-7.45978740000000,-17.7455900000000,-35.0679780000000,-36.5595700000000,-42.8227230000000,-47.7767980000000,-39.8632050000000,-16.2552600000000,-16.3998530000000,-3.28089710000000,19.1972270000000,41.7997860000000,64.8069310000000,81.0689390000000,81.6590580000000,68.3102260000000,55.8091320000000,31.3733630000000,-6.77967550000000,-27.1388490000000,-27.7729230000000,-27.7447510000000,-29.1105840000000,-22.8614880000000,-14.9392770000000,-10.5816430000000,-9.15556720000000,-7.29256110000000,-4.29435680000000,-11.3630680000000,-12.9076860000000,-10.3810070000000,-9.91120430000000,-18.6111740000000,-32.4139180000000,-37.2590450000000,-46.6695820000000,-54.0265920000000,-45.9827190000000,-21.7855470000000,-15.5881820000000,-2.47921370000000,14.3193750000000,28.7073060000000,53.5630040000000,75.1126480000000,76.4285810000000,63.1165470000000,51.6183130000000,27.2282520000000,-1.93500260000000,-23.5649090000000,-25.4652580000000,-25.6461700000000,-24.8643650000000,-19.2599260000000,-13.7381070000000,-10.5472870000000,-8.30998520000000,-6.26919840000000,-3.40063190000000,-9.54031660000000,-11.8440350000000,-12.3437230000000,-17.1045860000000,-25.4494090000000,-31.4973280000000,-35.3703880000000,-40.0440410000000,-45.4292070000000,-36.7789270000000,-15.8859800000000,-9.51868150000000,3.53041720000000,16.9855290000000,28.8762930000000,41.5089420000000,54.4187700000000,60.7834320000000,57.3923760000000,40.4930310000000,24.1973930000000,0.784138680000000,-16.9122660000000,-20.6451970000000,-21.6752930000000,-19.0458160000000,-14.5348150000000,-12.1781910000000,-9.93050480000000,-7.81605480000000,-4.79430060000000,-3.05638650000000,-9.52473640000000,-11.0146830000000,-12.6815480000000,-18.7947690000000,-24.0045150000000,-24.0714550000000,-25.2109150000000,-28.0692500000000,-27.3580860000000,-15.4939790000000,-5.44664050000000,-4.49710130000000,1.75524510000000,9.98222160000000,18.9431510000000,27.2405090000000,28.9828410000000,27.2420580000000,29.2845380000000,23.7602540000000,15.9755340000000,0.0407747780000000,-7.29834220000000,-10.4574100000000,-11.9232290000000,-11.3220920000000,-10.5015820000000,-9.64035420000000,-8.43842510000000,-6.24720570000000,-3.46545410000000,-2.13311000000000,-12.8962520000000,-10.4955570000000,-12.4670470000000,-15.2242190000000,-14.9557270000000,-13.6654680000000,-12.3242810000000,-12.5816130000000,-12.0865870000000,-7.79407120000000,-5.67843250000000,-4.59634830000000,-2.28561810000000,1.96948100000000,6.81114290000000,9.29467770000000,6.89549060000000,6.24796820000000,7.55959080000000,9.16065880000000,6.18247410000000,-3.84010290000000,-6.61799240000000,-6.42089130000000,-6.45520400000000,-7.04567000000000,-8.48626330000000,-8.88311580000000,-7.20925620000000,-4.18092490000000,-2.71172640000000,-0.961248340000000,-17.0661430000000,-13.4767400000000,-9.84325410000000,-10.9595600000000,-10.9828920000000,-10.4993930000000,-9.09317400000000,-8.04203890000000,-8.15170480000000,-8.70265960000000,-8.17000000000000,-6.16308210000000,-4.26158860000000,-3.85063770000000,-2.01097700000000,-2.32617120000000,-3.19388630000000,-2.68382050000000,-2.16359970000000,-2.02836320000000,-3.03126260000000,-6.00212380000000,-7.17367220000000,-7.07248500000000,-6.87153720000000,-6.32564310000000,-6.39227910000000,-7.56610200000000,-5.37092640000000,-3.35286880000000,-1.20289860000000,-0.957329870000000,-14.9657380000000,-18.0751060000000,-13.2681180000000,-9.63381860000000,-10.0524240000000,-9.91285320000000,-9.51101880000000,-8.47026820000000,-8.03125000000000,-8.30975440000000,-8.68280220000000,-7.98357960000000,-6.93127390000000,-6.18420700000000,-5.23241850000000,-5.17751410000000,-4.75860980000000,-3.64817190000000,-3.48813220000000,-3.98955610000000,-5.59427930000000,-6.85381410000000,-7.22919650000000,-7.38359450000000,-7.23559670000000,-6.88537360000000,-6.77703190000000,-5.08889290000000,-3.27389140000000,-1.59760900000000,-0.651357950000000,1.95736420000000,-1.88888760000000,-14.2683020000000,-17.2734870000000,-14.3211240000000,-10.2742080000000,-9.72518160000000,-9.08082680000000,-8.61122230000000,-8.41835020000000,-8.49451160000000,-8.52455810000000,-8.52779100000000,-7.96094270000000,-6.64172030000000,-5.14196780000000,-4.86573550000000,-4.59121990000000,-4.27322050000000,-4.29571440000000,-4.80593110000000,-5.70448490000000,-6.47942590000000,-7.23335270000000,-7.33023500000000,-7.54193210000000,-7.18882560000000,-5.29033040000000,-2.99292540000000,-1.20363090000000,-0.0121989900000000,3.44093470000000,16.1777000000000};
    float weights_2[len] = { 10.2837740000000,14.1613860000000,14.3763170000000,13.0077170000000,13.4239160000000,13.3877290000000,13.2317850000000,12.7474270000000,12.5000000000000,12.9900680000000,13.3169840000000,13.3999950000000,13.1745200000000,12.8382770000000,12.2287240000000,12.1171230000000,11.9764480000000,11.8249990000000,11.3955940000000,11.3741790000000,11.8379330000000,13.0896050000000,13.8532990000000,14.2699810000000,14.1563410000000,14.0994960000000,12.6806210000000,10.2472130000000,7.58101080000000,7.04316280000000,7.77740380000000,6.98634340000000,12.7324830000000,14.0016120000000,12.8387670000000,13.1263270000000,13.3188690000000,12.8024810000000,12.5058750000000,12.1605390000000,12.5648430000000,13.0964170000000,13.3083020000000,13.2905760000000,12.8463610000000,12.6189680000000,12.3183280000000,12.1565870000000,11.8975650000000,11.7463740000000,11.7354350000000,12.0361800000000,12.6703400000000,13.3411450000000,14.2235790000000,14.6100770000000,13.9136520000000,13.2768780000000,13.0224080000000,12.2939370000000,9.90734960000000,7.62295440000000,7.70978120000000,8.02865310000000,13.1609800000000,13.4670830000000,12.7494290000000,13.2435860000000,11.9601060000000,11.2032790000000,11.3595790000000,11.9142750000000,12.2271540000000,11.3825460000000,9.86742400000000,10.1482560000000,10.2328300000000,10.2203970000000,10.9350150000000,11.2268950000000,11.4550070000000,11.5847460000000,11.8184510000000,12.3075880000000,13.1361630000000,14.0470020000000,14.5157890000000,14.0787880000000,13.0177850000000,12.0037990000000,12.1139560000000,13.2243670000000,12.2783110000000,9.61167240000000,7.67923740000000,7.97768970000000,14.3324280000000,13.2258660000000,15.0236440000000,17.0610620000000,15.0328260000000,12.7288190000000,11.4150480000000,12.4219670000000,12.1404780000000,5.28770880000000,0.199812960000000,-0.547828380000000,-0.459883930000000,1.51868460000000,4.34345340000000,6.37234350000000,7.18683720000000,7.27584310000000,7.39548400000000,7.91467670000000,8.29474640000000,9.30838390000000,9.89228340000000,9.67325690000000,8.40146640000000,7.61000000000000,9.88630290000000,12.9608790000000,12.8786680000000,11.5685590000000,8.65286920000000,7.44480280000000,14.3762730000000,14.4537940000000,16.3141840000000,19.9674220000000,21.4795380000000,22.5366730000000,21.0307920000000,22.5227150000000,20.5612850000000,8.01568220000000,-7.36088320000000,-14.7720160000000,-16.9749770000000,-17.4361100000000,-13.9467700000000,-10.1490570000000,-7.76118660000000,-7.64366100000000,-2.35246370000000,-2.00430700000000,-2.77912550000000,-2.47178320000000,-0.397415970000000,0.536234680000000,-1.68021940000000,-2.15471790000000,1.06933930000000,6.52564430000000,8.43270970000000,9.93766120000000,9.77886870000000,6.55185840000000,14.0031470000000,15.4657730000000,16.5093270000000,20.7861600000000,30.1725460000000,45.3707540000000,43.2574310000000,43.6351090000000,40.8741190000000,23.0585500000000,-6.45393040000000,-23.9421440000000,-29.1256770000000,-33.4600110000000,-35.2636870000000,-35.6925930000000,-41.9807660000000,-39.7805520000000,-28.8508320000000,-20.7445910000000,-18.1619220000000,-16.4251310000000,-10.5053190000000,-5.57111170000000,-4.72403860000000,-8.28025630000000,-4.92553090000000,1.16285930000000,3.89893290000000,7.92767760000000,10.6100230000000,7.87260150000000,14.8420860000000,15.3893030000000,16.7068650000000,20.7532770000000,35.1876750000000,51.2661630000000,43.8651700000000,41.5062870000000,37.6278530000000,19.4232460000000,-9.58507440000000,-31.0622670000000,-36.1400380000000,-44.1804850000000,-53.0698170000000,-62.5458640000000,-73.1747820000000,-73.4282680000000,-69.8010180000000,-58.0686450000000,-43.8380090000000,-32.4831770000000,-25.0171110000000,-19.4547840000000,-16.6072560000000,-12.4908700000000,-6.25748350000000,0.253670750000000,3.53339840000000,9.28415970000000,11.9407520000000,10.6877420000000,15.3269290000000,14.7867820000000,16.7323020000000,22.1465680000000,36.8353310000000,51.9700010000000,42.3614690000000,34.0176930000000,27.6498600000000,12.3947940000000,-14.5119920000000,-36.2790030000000,-39.6556020000000,-45.6024700000000,-56.3396870000000,-75.9938740000000,-84.3914950000000,-89.6447530000000,-92.8211590000000,-86.8773650000000,-67.5718150000000,-46.1497610000000,-36.2952840000000,-29.4453700000000,-25.6365090000000,-18.6870690000000,-7.23882250000000,-0.208668860000000,4.15966840000000,10.9723230000000,12.7744880000000,12.2312210000000,14.4320660000000,13.9651770000000,15.9833410000000,19.6816900000000,33.3960760000000,51.2567210000000,39.1021800000000,25.6379450000000,20.4502810000000,11.5268770000000,-16.6936420000000,-34.8212550000000,-40.4495850000000,-50.0371060000000,-69.1392820000000,-82.5457610000000,-86.7216420000000,-93.9477010000000,-100.504530000000,-92.8134080000000,-70.1852570000000,-58.4642140000000,-46.5362010000000,-37.4459300000000,-29.8091720000000,-21.5261690000000,-9.80926610000000,-3.37789540000000,3.04707840000000,11.7778090000000,13.2955100000000,13.2267950000000,13.6365430000000,13.9150470000000,15.7927590000000,18.0496520000000,27.6145190000000,48.1909750000000,33.8587300000000,18.1000000000000,19.6510940000000,13.3747150000000,-14.4766430000000,-32.4838560000000,-36.5672720000000,-55.2411160000000,-79.8614580000000,-87.3849720000000,-90.5353620000000,-93.8388520000000,-95.9911730000000,-81.7319030000000,-75.5238800000000,-64.1979680000000,-53.2140200000000,-46.0686840000000,-40.6617550000000,-29.8037260000000,-12.9080400000000,-6.68171360000000,2.59448270000000,11.7270230000000,13.4377510000000,14.0285250000000,13.4692970000000,13.8642030000000,14.0766530000000,16.1771530000000,24.0998740000000,44.1434480000000,23.3993740000000,9.38712120000000,15.1895790000000,12.3749270000000,-6.80187890000000,-16.2602960000000,-13.7547050000000,-36.9634860000000,-53.6800500000000,-71.0789340000000,-87.6187360000000,-93.1874920000000,-84.1494900000000,-75.9420390000000,-75.4221880000000,-58.8100280000000,-44.6126330000000,-39.7372700000000,-45.7532230000000,-42.6739880000000,-24.5565490000000,-12.7665440000000,2.51122360000000,10.6932870000000,13.1143470000000,14.0030060000000,12.5768370000000,12.3732810000000,12.6642470000000,15.4317210000000,21.4178920000000,34.5029790000000,14.9065480000000,6.53622870000000,10.5961030000000,5.90240050000000,-4.42028860000000,1.39624070000000,2.53523370000000,-3.78529620000000,-9.17863940000000,-28.6123700000000,-70.0334700000000,-85.9672010000000,-74.3399890000000,-72.7190020000000,-68.5981370000000,-44.1457940000000,-22.8127060000000,-22.3479270000000,-28.6132470000000,-24.8994670000000,-23.4062460000000,-23.0769460000000,-2.28235910000000,7.64537760000000,12.3302550000000,14.0488120000000,11.8790700000000,12.0310680000000,12.2260740000000,14.5381740000000,18.0819400000000,22.7708450000000,11.1013970000000,2.21623640000000,7.08333730000000,6.55737450000000,1.48696350000000,10.1569300000000,12.1247230000000,18.9243140000000,18.7102510000000,7.98645540000000,-31.2244090000000,-48.6026920000000,-56.4394490000000,-58.9152760000000,-52.7861180000000,-25.7044790000000,-7.58943800000000,-16.2447410000000,-14.7022850000000,-10.5262340000000,-12.2406740000000,-19.8261550000000,-5.81599520000000,5.57053610000000,11.2764640000000,13.6909880000000,11.9865120000000,12.2699090000000,12.3163840000000,13.7748600000000,15.0268840000000,11.5784830000000,3.44967680000000,0.0930167960000000,7.08594180000000,15.1672960000000,23.0654450000000,41.1652600000000,54.1513860000000,58.3800050000000,59.3157460000000,48.9006500000000,18.5273740000000,4.35018630000000,-16.8421990000000,-24.4328350000000,-23.4946920000000,-2.66713830000000,6.89092450000000,3.09322450000000,5.94000000000000,9.27630420000000,5.60800220000000,0.509698210000000,6.24733020000000,8.83176330000000,11.8353360000000,13.6447700000000,11.8374400000000,12.5665250000000,12.3573050000000,11.9186910000000,9.04329400000000,-2.16913010000000,-16.0146680000000,-8.38245680000000,7.84706020000000,13.7340350000000,27.4963460000000,62.2690280000000,80.3404010000000,87.9050980000000,91.0838850000000,84.5306320000000,56.7970050000000,28.3269290000000,15.1173250000000,6.22990320000000,-0.197539930000000,6.02098560000000,24.4246250000000,25.5623250000000,23.3424550000000,21.5401100000000,20.1074370000000,17.1570550000000,9.54953480000000,11.1882820000000,13.1198050000000,13.2955970000000,11.7260250000000,12.6189120000000,12.2697950000000,9.01444720000000,-3.63298730000000,-11.9344230000000,-21.7542690000000,-18.3139400000000,-4.66338250000000,6.44606780000000,22.3038230000000,69.8289490000000,90.4398960000000,98.2211300000000,106.180400000000,113.062020000000,68.7969130000000,35.7874570000000,26.8336830000000,17.8938790000000,4.84552240000000,9.48792460000000,42.4476930000000,45.5493580000000,41.3344920000000,35.2450940000000,39.4678610000000,27.6283400000000,13.1098510000000,11.8895610000000,13.4919950000000,13.1123030000000,12.0674810000000,12.1007000000000,10.7451270000000,2.65383220000000,-16.3040490000000,-17.8457160000000,-22.0294400000000,-18.6235900000000,-12.2592320000000,-2.72780900000000,25.6123560000000,76.2666020000000,92.4157030000000,96.1607970000000,111.188640000000,118.782880000000,64.5237120000000,38.9985960000000,27.1557980000000,16.4630600000000,-7.94099190000000,10.3685870000000,47.3533020000000,49.8899270000000,44.6923560000000,47.3739470000000,54.5145910000000,35.0107880000000,18.0399280000000,14.7038600000000,13.1286330000000,12.5023480000000,11.9783130000000,10.9979460000000,8.69990350000000,-2.72945930000000,-20.8044490000000,-19.5667210000000,-21.0513900000000,-18.7746370000000,-10.9150810000000,1.13196340000000,33.6162640000000,74.3004380000000,85.9086760000000,95.2137680000000,112.094890000000,100.218870000000,49.7768170000000,35.1996540000000,26.1572210000000,4.39917140000000,-16.3011040000000,13.9668050000000,48.0250850000000,46.8770260000000,42.6512680000000,59.2274550000000,59.5564270000000,39.5980110000000,19.9403130000000,15.4193530000000,13.3243910000000,11.7963590000000,10.9624230000000,9.48668960000000,6.32013080000000,-0.972464620000000,-11.8870150000000,-16.3257520000000,-20.0207180000000,-16.9438020000000,-11.2247650000000,10.7722140000000,32.6012190000000,57.6302720000000,64.9857710000000,77.8245470000000,91.3040540000000,66.2887570000000,26.3120750000000,15.3149810000000,2.73792790000000,-17.9188420000000,-14.5039920000000,19.9619100000000,47.6473120000000,42.4196170000000,47.8348120000000,60.3872910000000,55.4871180000000,41.2505870000000,22.6344720000000,15.5654540000000,13.5359270000000,11.2196230000000,10.8604540000000,8.21347620000000,3.67201760000000,-2.78264620000000,-6.18782620000000,-0.745680030000000,-9.83817480000000,-13.8059040000000,-8.15770630000000,9.99706460000000,10.7491740000000,15.9271830000000,18.5813390000000,24.6087090000000,25.0815110000000,8.52212240000000,-13.2554780000000,-20.5762020000000,-41.9804120000000,-51.8021660000000,-32.3000000000000,7.08242890000000,32.1787720000000,32.7277910000000,45.1227570000000,48.2209130000000,43.7780840000000,34.5727500000000,21.6253850000000,15.8118230000000,13.4100870000000,10.6865880000000,12.0299840000000,7.79118970000000,0.692912580000000,-3.64941170000000,-4.21168470000000,6.55056480000000,2.55072160000000,2.30968280000000,3.75069900000000,6.74491790000000,-0.879918580000000,-1.89267600000000,-3.23968550000000,-5.34218550000000,-21.3298130000000,-43.7200890000000,-56.9957310000000,-59.2162970000000,-74.8835680000000,-72.0189290000000,-46.5534290000000,-12.1548790000000,14.8837260000000,33.0069050000000,36.8474920000000,36.2644270000000,29.4636750000000,20.5837080000000,12.9908280000000,11.2513550000000,9.98903850000000,8.58993430000000,11.7460970000000,7.96050020000000,0.397853790000000,-3.67889140000000,0.104458440000000,13.9273220000000,10.2745250000000,18.5588230000000,27.9855160000000,24.5518970000000,-0.575730620000000,1.56280970000000,-0.879778560000000,-16.9290490000000,-37.9477650000000,-59.2108420000000,-79.1832660000000,-82.6579970000000,-77.7122960000000,-61.3377150000000,-41.9803960000000,-15.0935660000000,19.9419350000000,33.5584950000000,31.7122330000000,33.8359180000000,27.9828190000000,16.6697810000000,11.1958680000000,8.93781570000000,7.58006000000000,6.82451490000000,12.4265920000000,10.0237740000000,4.88542460000000,-0.157064330000000,4.70435290000000,21.3764820000000,20.7676450000000,30.1975800000000,39.2554020000000,42.1389770000000,15.4937280000000,5.94321820000000,-7.47806360000000,-26.7610650000000,-43.3322870000000,-63.9307100000000,-84.3551250000000,-88.9747770000000,-75.9448090000000,-56.4794200000000,-40.1593060000000,-7.26288750000000,27.7221890000000,30.1693480000000,27.9354130000000,29.4119630000000,24.5448340000000,15.8926500000000,11.4262720000000,8.13373280000000,6.55839300000000,5.89897680000000,12.5898000000000,12.4392450000000,9.21194270000000,5.34528730000000,11.9103290000000,28.1791310000000,27.9910980000000,36.0327380000000,42.7167740000000,40.0831530000000,18.6920970000000,17.6828120000000,-3.69080710000000,-33.6808550000000,-48.4417990000000,-65.2402110000000,-83.2719800000000,-86.5283200000000,-73.9540860000000,-56.4622230000000,-37.4560700000000,2.23862700000000,27.9884430000000,27.9976790000000,27.2737410000000,30.0644380000000,23.9975240000000,15.5229580000000,11.8293210000000,9.19418330000000,6.34416480000000,4.83159920000000,12.3772750000000,12.7402630000000,9.87449840000000,7.47449540000000,17.7421040000000,35.0599330000000,36.5802500000000,42.8245050000000,47.7641560000000,39.8595920000000,16.2801930000000,16.4092080000000,3.26817770000000,-19.2195530000000,-41.7946700000000,-64.7754060000000,-81.0595630000000,-81.6691060000000,-68.3204880000000,-55.8202400000000,-31.3839820000000,6.76938490000000,27.1413210000000,27.7730690000000,27.7129900000000,29.1049160000000,22.8513470000000,14.9387540000000,10.5824580000000,9.16718010000000,7.29885480000000,4.29230880000000,11.3828510000000,12.8951670000000,10.3775500000000,9.89312740000000,18.6250820000000,32.4062500000000,37.2461360000000,46.6513900000000,54.0042760000000,45.9661520000000,21.7816220000000,15.5549800000000,2.47140960000000,-14.3293870000000,-28.7243880000000,-53.5669820000000,-75.1126330000000,-76.4360730000000,-63.0937190000000,-51.6128690000000,-27.2405680000000,1.92055450000000,23.5804790000000,25.4760380000000,25.6519130000000,24.8996390000000,19.2535190000000,13.7434380000000,10.5633250000000,8.33109950000000,6.25661800000000,3.40351220000000,9.52495380000000,11.8333490000000,12.3165460000000,17.0771920000000,25.4340380000000,31.4915620000000,35.3000000000000,40.0481570000000,45.4239500000000,36.7727620000000,15.8738110000000,9.51003930000000,-3.53237580000000,-16.9681070000000,-28.8916740000000,-41.5126570000000,-54.4361920000000,-60.7976990000000,-57.3807220000000,-40.4742810000000,-24.2016240000000,-0.783895490000000,16.9249760000000,20.6513200000000,21.6697120000000,19.0365240000000,14.5397150000000,12.1720690000000,9.91145990000000,7.82137780000000,4.80142930000000,3.04299090000000,9.52148910000000,10.9966580000000,12.6715530000000,18.7955380000000,24,24.0571690000000,25.1925750000000,28.0697190000000,27.3733520000000,15.5039080000000,5.46359210000000,4.51972390000000,-1.78795670000000,-9.97561550000000,-18.9471110000000,-27.2571600000000,-28.9656790000000,-27.2383290000000,-29.2780610000000,-23.7486530000000,-15.9593900000000,-0.0108372170000000,7.30993030000000,10.4487830000000,11.9298020000000,11.2994830000000,10.5231750000000,9.63160710000000,8.46107010000000,6.23756980000000,3.46071980000000,2.13181660000000,12.8894050000000,10.4982070000000,12.4703130000000,15.2321800000000,14.9283580000000,13.6539030000000,12.3180120000000,12.5941170000000,12.0910610000000,7.83345220000000,5.68214130000000,4.58921150000000,2.28040860000000,-2.00535940000000,-6.82060000000000,-9.29824920000000,-6.88542800000000,-6.24132540000000,-7.56731890000000,-9.15513320000000,-6.17510890000000,3.84766940000000,6.63155080000000,6.42785980000000,6.47785710000000,7.02919580000000,8.50138660000000,8.86351780000000,7.21616510000000,4.18753580000000,2.74385170000000,0.969324710000000,17.0663620000000,13.4905410000000,9.83576300000000,10.9490280000000,10.9863960000000,10.5029060000000,9.10624500000000,8.03000000000000,8.14417360000000,8.68930340000000,8.17755130000000,6.14588590000000,4.26149940000000,3.84120250000000,2.03573040000000,2.30841470000000,3.20237330000000,2.68085170000000,2.17035910000000,2.02230310000000,3.01737000000000,6.01967480000000,7.21189590000000,7.08021020000000,6.90117030000000,6.32115080000000,6.37829920000000,7.54543920000000,5.36723760000000,3.34208060000000,1.20268730000000,0.933908280000000,14.9612820000000,18.0589960000000,13.2747200000000,9.62541580000000,10.0532400000000,9.92709160000000,9.51498220000000,8.48373600000000,8.05252460000000,8.29659940000000,8.70516970000000,7.97391370000000,6.94226500000000,6.15650130000000,5.22472620000000,5.18840220000000,4.75684120000000,3.67117050000000,3.48576620000000,3.98639180000000,5.57331320000000,6.85420510000000,7.22193480000000,7.37398340000000,7.23027130000000,6.86420440000000,6.77000470000000,5.09116360000000,3.28663710000000,1.59136310000000,0.630554800000000,-1.94399860000000,1.89749380000000,14.2455320000000,17.2847040000000,14.3251700000000,10.2928740000000,9.73248100000000,9.06951330000000,8.61339470000000,8.42289730000000,8.49571040000000,8.52733800000000,8.54121490000000,7.94424060000000,6.64597800000000,5.13694290000000,4.87742040000000,4.60199980000000,4.28043130000000,4.30517340000000,4.82000000000000,5.73036860000000,6.47360230000000,7.20913740000000,7.31989000000000,7.54323670000000,7.20962380000000,5.27913760000000,2.98201890000000,1.21234930000000,0.0179577540000000,-3.46345880000000,-16.1797490000000 };

    for (int i = 0; i < len; i++)
    {
        poss_F = poss_F + weights_1[i] * image[i];
        poss_T = poss_T + weights_2[i] * image[i];
    }

    poss[0] = 1 / (1 + exp(- poss_F / scale));
    poss[1] = 1 / (1 + exp(- poss_T / scale));
}

*/