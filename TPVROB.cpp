#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

vector<Point2f> X;
vector<Point3f> X_3D;
vector<vector<Point3f>> objPoint;
vector<vector<Point2f>> imgPoint;

Mat K, distCoeffs;
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
    X_3D.push_back(Point3f(1.0f, 0.0f, 0.0f));
    X_3D.push_back(Point3f(0.0f, 1.0f, 0.0f));
    X_3D.push_back(Point3f(1.0f, 1.0f, 0.0f));
    objPoint.push_back(X_3D);

    vector<Point2f> objectPointsPlanar;
    objectPointsPlanar.push_back(Point2f(0.0f, 0.0f));
    objectPointsPlanar.push_back(Point2f(1.0f, 0.0f));
    objectPointsPlanar.push_back(Point2f(0.0f, 1.0f));
    objectPointsPlanar.push_back(Point2f(1.0f, 1.0f));

    calibrateCamera(objPoint, imgPoint, I0.size(), K, distCoeffs, R, T);

    cout << "-------------------" << endl;
    vector<Mat> corners;

    for (const auto& point : X)
    {
        Mat vectX = Mat(3, 1, CV_32F, 0.0f);
        vectX.at<float>(0, 0) = point.x;
        vectX.at<float>(1, 0) = point.y;
        vectX.at<float>(2, 0) = 1.0f;
        vectX.convertTo(vectX, K.type());
        cout << "point clique : \n" << vectX << endl;
        corners.push_back(vectX);
    }

    /*vector<Mat>cornersMeter;
    for (const auto& p : corners)
    {
        Mat Xmeter = K.inv() * p;
        cout << "Point repere camera : \n"<< Xmeter << endl;
        cornersMeter.push_back(Xmeter);
    }*/

    Mat H = findHomography(objectPointsPlanar, X);

    //------------------------------------------------------------------------------

    Mat projRT, Rmat, r12, r1, r2, PI;
    Rodrigues(R[0], Rmat);
    hconcat(Rmat.col(0), Rmat.col(1), r12);
    hconcat(r12, T[0], projRT);

    Mat d = Mat::eye(3, 3, projRT.type());
    Mat h12 = K.inv() * H;
    int s = h12.col(2).rows / h12.col(1).rows;
    d.at<double>(1, 1) = 1.0 / s;

    Mat H0w = H * d;
    Mat toR = K.inv() * H0w;
    r1 = toR.col(0); r2 = toR.col(1);

    Mat tmp1, tmp2;
    hconcat(r1, r2, tmp1);
    hconcat(tmp1, r1.cross(r2), tmp2);
    hconcat(tmp2, h12.col(2), PI);

    Mat P0w = K * PI;
    cout << "----------------" << endl;
    Mat homogene = Mat(1, 4, P0w.type(), 0.0);
    homogene.at<double>(0, 3) = 1.0;
    vconcat(P0w, homogene, P0w);
    cout << P0w << endl;
    vector<Mat>xtest;
    for (const auto& point : X)
    {
        Mat vectX = Mat(4, 1, P0w.type(), 0.0);
        vectX.at<double>(0, 0) = point.x;
        vectX.at<double>(1, 0) = point.y;
        vectX.at<double>(2, 0) = 0.0;
        vectX.at<double>(3, 0) = 1.0;
        xtest.push_back(vectX);
    }
    for (const auto& x : xtest)
    {
        cout << "Point en px : \n" << x << endl;
        Mat posetest = P0w.inv() * x - Mat(4, 1, P0w.type(), 1.0);
        cout << "Point en 3D : \n" << posetest << endl;
    }

    vector<Mat>pointTest;
    for (const auto& p : X_3D)
    {
        Mat vec = Mat(4, 1, P0w.type(), 0.0);
        vec.at<double>(0, 0) = p.x;
        vec.at<double>(1, 0) = p.y;
        vec.at<double>(2, 0) = 0.0;
        vec.at<double>(3, 0) = 1.0;

        Mat posetest = P0w * vec;

        cout << "Point en px : \n" << posetest << endl;
        pointTest.push_back(posetest);
    }
    Mat I0_ = I0.clone();
    line(I0_, Point(pointTest[0].at<double>(0, 0), pointTest[0].at<double>(1, 0)), Point(pointTest[1].at<double>(0, 0), pointTest[1].at<double>(1, 0)), Scalar(0, 0, 255), 2);
    line(I0_, Point(pointTest[0].at<double>(0, 0), pointTest[0].at<double>(1, 0)), Point(pointTest[2].at<double>(0, 0), pointTest[2].at<double>(1, 0)), Scalar(0, 255, 0), 2);
    line(I0_, Point(pointTest[0].at<double>(0, 0), pointTest[0].at<double>(1, 0)), Point(pointTest[3].at<double>(0, 0), pointTest[3].at<double>(1, 0)), Scalar(255, 0, 0), 2);
    imshow("Corners", I0_);
    cv::waitKey(0);

    //------------------------------------------------------------------------------

    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> keyPoints;
    Mat kpMask = Mat::zeros(I0.rows, I0.cols, CV_8U);
    for (int j = X[0].x; j < X[3].x + 1; j++)
    {
        for (int i = X[0].y; i < X[3].y + 1; i++)
        {
            kpMask.at<uint8_t>(i, j) = 1;
        }
    }
    sift->detect(I0, keyPoints, kpMask);
    Mat img = I0.clone();
    drawKeypoints(I0, keyPoints, img);
    imshow("KeyPoints", img);
    cv::waitKey(0);
}



