#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

vector<Point2f> X;
vector<Point3f> X_3D;
vector<vector<Point3f>> objPoint;
vector<vector<Point2f>> imgPoint;

Mat cameraMatrix, distCoeffs;
vector<Mat> R, T;

void onMouse(int action, int x, int y, int, void*) {
    if (action == cv::EVENT_LBUTTONDOWN) {
        X.push_back(Point{ x, y });
    }
}

string openCVType2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

int main(int argc, char** argv)
{
    string smallPath = R"(C:\Users\user\Documents\_ESIRTP\3\VROB\imtest.jpg)";
    string bigPath = R"(D:\Documents\_TPESIR\3\VROB_data\imtest.jpg)";

    Mat I0full = imread(smallPath);
    Mat I0;

    resize(I0full, I0, Size(I0full.cols / 2, I0full.rows / 2)); //width, height

    //----------------------------------------------------------------------------------

    namedWindow("Corners Select");

    setMouseCallback("Corners Select", onMouse);

    while (X.size() < 4) {
        for (const auto& point : X) {
            circle(I0, point, 2, Scalar{ 0, 0, 255 }, FILLED);
        }
        imshow("Corners Select", I0);
        cv::waitKey(25);
    }

    //------------------------------------------------------------------------------

    imgPoint.push_back(X);

    X_3D.push_back(Point3f(0.0f, 0.0f, 0.0f));
    X_3D.push_back(Point3f(0.21f, 0.0f, 0.0f));
    X_3D.push_back(Point3f(0.0f, 0.16f, 0.0f));
    X_3D.push_back(Point3f(0.21f, 0.16f, 0.0f));
    objPoint.push_back(X_3D);

    calibrateCamera(objPoint, imgPoint, I0.size(), cameraMatrix, distCoeffs, R, T);

    vector<Mat> corners;
    for (const auto& point : X)
    {
        Mat vectX = Mat(3, 1, CV_32F, 0.0f);
        vectX.at<float>(0, 0) = point.x;
        vectX.at<float>(1, 0) = point.y;
        vectX.at<float>(2, 0) = 1.0f;
        vectX.convertTo(vectX, cameraMatrix.type());
        cout << vectX << endl;
        corners.push_back(vectX);
    }

    vector<Mat>cornersMeter;
    for (const auto& p : corners)
    {
        Mat Xmeter = cameraMatrix.inv() * p;
        cout << Xmeter << endl;
        cornersMeter.push_back(Xmeter);
    }
    for (const auto& p : cornersMeter) {
        Mat px = cameraMatrix * p;
        px.convertTo(px, CV_32SC1);
        cout << px << endl;
        Point point = Point{ px.at<int>(0,0), px.at<int>(1, 0)};
        circle(I0, point, 5, Scalar{ 255, 0, 0 }, FILLED);
    }
    imshow("Corners", I0);
    cv::waitKey(0);
}



