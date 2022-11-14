#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

vector<Point2f> X;
vector<Point3f> X_3D;
vector<vector<Point3f>> objPoint;
vector<vector<Point2f>> imgPoint;

Mat cameraMatrix, distCoeffs, R, T;

void onMouse(int action, int x, int y, int, void*) {
    if (action == cv::EVENT_LBUTTONDOWN) {
        X.push_back(Point{ x, y });
    }
}

int main(int argc, char** argv)
{
    string smallPath = R"(C:\Users\user\Documents\_ESIRTP\3\VROB\imtest.jpg)";
    string bigPath = R"(D:\Documents\_TPESIR\3\VROB_data\imtest.jpg)";

    Mat I0full = imread(bigPath);
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
        waitKey(25);
    }

    //------------------------------------------------------------------------------

    imgPoint.push_back(X);
    Mat test = Mat(3,1,CV_32F, 0.0f);
    for (const auto& point : X)
    {
        cout << point.x << "," << point.y << endl;
        X_3D.push_back(Point3f(point.x, point.y, 0.0f));
        test.at<float>(0, 0) = point.x;
        test.at<float>(1, 0) = point.y;
        test.at<float>(1, 0) = 1.0f;
    }
    objPoint.push_back(X_3D);

    calibrateCamera(objPoint, imgPoint, I0.size(), cameraMatrix, distCoeffs, R, T);

    test.convertTo(test, cameraMatrix.type());

    Mat Xmeter = cameraMatrix.inv() * test;

    cout << Xmeter << endl;
}



