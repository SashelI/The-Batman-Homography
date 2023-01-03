#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;

vector<Point2f> X;                  //Vecteur des coins cliqués en pixels
vector<Point3f> X_Meter_3D;         //vecteur des coins en 3D monde (origine en haut à gauche)
vector<vector<Point3f>> objPoint;   //Vecteur des points 3D pour opencv
vector<vector<Point2f>> imgPoint;   //Vecteur des points pixels pour opencv

Mat K, distCoeffs;
vector<Mat> R, T;

Mat Homographies;

/**
 * Enregistre les coordonnees en pixel du point clique a la souris dans le vecteur X
 */
void onMouse(int action, int x, int y, int, void*) {
    if (action == cv::EVENT_LBUTTONDOWN) {
        X.push_back(Point{ x, y });
    }
}

/**
 * Utilitaire
 */
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

/**
 * calcul de pose a partir de l'homographie selon la methode decrite dans l'etude source
 * @param Hiw : Homographie de monde a pixel frame i
 * @return Pose de la camera correspondant a l'homographie.
 */
Mat findPose(Mat& Hiw)
{
    Mat r1, r2, PI;                     //Matrices colonnes de rotation et pose temporaire
    Mat HtoR = K.inv() * Hiw;           //D'homographie à vecteurs rotation et translation
    r1 = HtoR.col(0); r2 = HtoR.col(1);

    Mat tmp1, tmp2;
    hconcat(r1, r2, tmp1);
    hconcat(tmp1, r1.cross(r2), tmp2);
    hconcat(tmp2, HtoR.col(2), PI);     //PI = [r1 r2 r1xr2 t]

    Mat P0w = K * PI;                   //Application de la matrice intrinseque K
    Mat homogene = Mat(1, 4, P0w.type(), 0.0);  //Ajout d'une ligne [0 0 0 1] pour matrice de pose carree
    homogene.at<double>(0, 3) = 1.0;
    vconcat(P0w, homogene, P0w);
    return P0w;
}

/**
 * Ajoute l'image 0 dans un coin de la video pour rendu
 * @param frame : frame de la video
 * @param img : image a superposer
 * @param x : position x souhaitee de l'image
 * @param y : position y souhaitee de l'image
 */
void addImg(Mat& frame, const Mat& img, int& x, int& y)
{
    for (int i = x; i < x + img.cols; i++)
    {
        for (int j = y; j < y + img.rows; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                frame.at<Vec3b>(j, i)[k] = img.at<Vec3b>(j - y, i - x)[k];
            }
        }
    }
}

/**
 * Calcule les correspondances entre les coins cliques dans l'image 0 et la frame courante.
 * Dessine une ligne de suivi reliant les coins de l'image 0 et les coins dans la frame courante
 * @param frame : frame courante de la video
 * @param c0 : coins de l'image 0
 * @param s : échelle de l'image
 * @param x : position x de l'image
 * @param y : position y de l'image
 * @param H : homographie entre l'image 0 et la frame courante
 */
void drawCorners(Mat& frame, const vector<Mat>& c0, int& s, int& x, int& y, Mat H)
{
    vector<Point2f>corners1;
    vector<Point2f>corners0;
    for(auto& p : c0)
    {
        Mat p1 = H * p;
        corners1.push_back(Point2f(p1.at<double>(0,0)/ p1.at<double>(2,0), p1.at<double>(1,0)/ p1.at<double>(2,0)));
        corners0.push_back(Point2f(p.at<double>(0, 0), p.at<double>(1, 0)));
    }
    for (int i = 0; i < corners1.size(); i++)
    {
        Point2f p0 = Point2f(corners0[i].x / s + x, corners0[i].y / s + y);
        circle(frame, corners1[i], 2, Scalar(0, 0, 255), FILLED);
        line(frame, p0, corners1[i], Scalar(255, 213, 0));
    }
}

int main(int argc, char** argv)
{
    //PENSER A CHANGER LES PATH AVANT RENDU
    string smallPath = R"(C:\Users\user\Documents\_ESIRTP\3\VROB\Batman_r.jpg)";
    string bigPath = R"(D:\Documents\_TPESIR\3\VROB_data\Batman_r.jpg)";
    string smallVideo = R"(C:\Users\user\Documents\_ESIRTP\3\VROB\BatVideo.mp4)";
    string bigVideo = R"(D:\Documents\_TPESIR\3\VROB_data\BatVideo.mp4)";

    Mat I0full = imread(smallPath);
    Mat I0;
    resize(I0full, I0, Size(I0full.cols/1.5, I0full.rows / 1.5)); //width, height

	//----------------------------------------------------------------------------------
    // Selection des coins a la souris
    //----------------------------------------------------------------------------------
    {cout <<
        "----------------------------------------------------------------------------------\n Selection des coins a la souris \n ----------------------------------------------------------------------------------"
        << endl; }

    namedWindow("Corners Select");

    setMouseCallback("Corners Select", onMouse);

    while (X.size() < 4) {
        for (const auto& point : X) {
            circle(I0, point, 2, Scalar{ 0, 0, 255 }, FILLED);
        }
        imshow("Corners Select", I0);
        cv::waitKey(25);
    }
    circle(I0, X[3], 2, Scalar{ 0, 0, 255 }, FILLED);

    //----------------------------------------------------------------------------------
    // Creation des vecteurs de points et calibration
    //----------------------------------------------------------------------------------
    {cout <<
        "----------------------------------------------------------------------------------\n Creation des vecteurs de points et calibration \n ----------------------------------------------------------------------------------"
        << endl; }

    imgPoint.push_back(X);

    X_Meter_3D.push_back(Point3f(0.0f, 0.0f, 0.0f));
    X_Meter_3D.push_back(Point3f(1.0f, 0.0f, 0.0f));
    X_Meter_3D.push_back(Point3f(0.0f, 1.0f, 0.0f));
    X_Meter_3D.push_back(Point3f(1.0f, 1.0f, 0.0f));
    objPoint.push_back(X_Meter_3D);

    vector<Point2f> objectPointsPlanar;                 //coins monde en 2D pour homographie
    objectPointsPlanar.push_back(Point2f(0.0f, 0.0f));
    objectPointsPlanar.push_back(Point2f(1.0f, 0.0f));
    objectPointsPlanar.push_back(Point2f(0.0f, 1.0f));
    objectPointsPlanar.push_back(Point2f(1.0f, 1.0f));

    calibrateCamera(objPoint, imgPoint, I0.size(), K, distCoeffs, R, T);    //Calibration de la camera. K : matrice intrinseque
    cout << "Matrice de paramètres intrinsèques caméra : \n" << K << endl;
    cout << "-------------------" << endl;

    vector<Mat> corners;        //vecteur des coins sous forme de matrices en coordonnees homogenes, pour traitements
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

    vector<Point2f>cornersMeter;
    for (const auto& p : corners)
    {
        Mat Xmeter = K.inv() * p;
        cout << "Point repere camera : \n"<< Xmeter << endl;
        cornersMeter.push_back(Point2f(Xmeter.at<double>(0,0), Xmeter.at<double>(1,0)));
    }

    //----------------------------------------------------------------------------------
    // Homographie zero et pose zero | calcul du repere zero
    //----------------------------------------------------------------------------------
    {cout <<
        "----------------------------------------------------------------------------------\n Homographie zero et pose zero | calcul du repere zero \n ----------------------------------------------------------------------------------"
        << endl; }

    Mat H = findHomography(objectPointsPlanar, X);  //Homograpghie entre les points monde et camera de l'image 0

    Mat d = Mat::eye(3, 3, K.type());
    Mat h12 = K.inv() * H;
    int s = h12.col(2).rows / h12.col(1).rows;
    d.at<double>(1, 1) = 1.0 / s;                   //matrice pour correction d'echelle

    Mat H0w = H * d;                                //correction d'echelle

    Mat P0w = findPose(H0w);                        //calcul de la pose a partir de l'homographie
    cout << "Pose world to zero : \n"<< P0w << endl;
    cout << "-------------------" << endl;
    vector<Mat>xEn3D;
    for (const auto& point : X)
    {
        Mat vectX = Mat(3, 1, P0w.type(), 0.0);
        vectX.at<double>(0, 0) = point.x;
        vectX.at<double>(1, 0) = point.y;
        vectX.at<double>(2, 0) = 0.0;
        xEn3D.push_back(vectX);
    }
    for (const auto& x : xEn3D)
    {
        cout << "Point en px : \n" << x << endl;
        Mat posetest = H0w.inv() * x;
        cout << "Point en 3D : \n" << posetest << endl;
    }

    Mat axisx = Mat({ 1.0, 0.0, 0.0, 1.0 });        //Points du repere monde en 3D homogene
    Mat axisy = Mat({ 0.0, 1.0, 0.0, 1.0 });
    Mat axisz = Mat({ 0.0, 0.0, 1.0, 1.0 });
    Mat origin = Mat({ 0.0, 0.0, 0.0, 1.0 });

    vector<Mat> coordinateSystem = { origin, axisx, axisy, axisz };

    vector <Point2f> coordSystem2D;                 //Vecteur des points image du repere en 2D
    for (const auto& p : coordinateSystem)
    {
        Mat tmp = P0w * p;
        Point2f cs2D = Point2f(tmp.at<double>(0, 0) / tmp.at<double>(2, 0), tmp.at<double>(1, 0) / tmp.at<double>(2, 0));
        coordSystem2D.push_back(cs2D);
    }
    Mat I0_axis = I0.clone();
    line(I0_axis, coordSystem2D[0], coordSystem2D[1], Scalar(0, 0, 255), 2, LINE_AA);   //Tracage des axes sur l'image
    line(I0_axis, coordSystem2D[0], coordSystem2D[2], Scalar(0, 255, 0), 2, LINE_AA);
    line(I0_axis, coordSystem2D[0], coordSystem2D[3], Scalar(255, 0, 0), 2, LINE_AA);
    cout << ":::::::::::::::Appuyez sur une touche pour continuer:::::::::::::::" << endl;
    imshow("Corners", I0_axis);
    cv::waitKey(0);

    //----------------------------------------------------------------------------------
    // Calcul des points d'interets sur l'image 0
    //----------------------------------------------------------------------------------
    {cout <<
        "----------------------------------------------------------------------------------\n Calcul des points d'interets sur l'image 0 \n ----------------------------------------------------------------------------------"
        << endl; }

    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> keyPoints0;                        //Points cles
    Mat desc0, desc1;                                   //Descripteurs
    Mat kpMask = Mat::zeros(I0.rows, I0.cols, CV_8U);   //Masque de detection
    for (int j = X[0].x; j < X[3].x + 1; j++)           //On ne souhaite detecter les points cles qu'entre les quatre coins choisis
    {
        for (int i = X[0].y; i < X[3].y + 1; i++)
        {
            kpMask.at<uint8_t>(i, j) = 1;
        }
    }
    sift->detectAndCompute(I0, kpMask, keyPoints0, desc0);  //Detection des points sift
    Mat img = I0.clone();
    drawKeypoints(I0, keyPoints0, img, Scalar(51, 255, 255));
    int imgScale = 2;
    resize(img, img, Size(img.cols / imgScale, img.rows / imgScale));
    /*cv::imshow("KeyPoints", img);
    cv::waitKey(0);*/
    int x = I0full.cols - img.cols; //position x pour rendu
    int y = 0;                      //position y pour rendu

	//----------------------------------------------------------------------------------
    // Calcul de pose et axes sur la premiere frame de la video
    //----------------------------------------------------------------------------------
    {cout <<
        "----------------------------------------------------------------------------------\n Calcul de pose et axes sur la premiere frame de la video \n ----------------------------------------------------------------------------------"
        << endl; }

	VideoWriter render("FinalRender10f.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(I0full.cols, I0full.rows));
    VideoCapture vid;
    vid.open(smallVideo);

    vector<Point2f>kp0, newkp, oldkp, kp0H, kp1H;

    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();   //matcher Flann

    vector<KeyPoint> keyPoints1;
    vector<vector<DMatch>> matches0;
    Mat frame1;
    vector<DMatch> goodMatches;

    vid.read(frame1);
    //resize(frame1, frame1, Size(frame1.cols / 1.5, frame1.rows / 1.5));

    sift->detectAndCompute(frame1, Mat(), keyPoints1, desc1);
    matcher->knnMatch(desc0, desc1, matches0, 2);               //match des points cles entre l'image 0 et la frame 1 avec flann

    for (unsigned int i = 0; i < matches0.size(); ++i) {
        if (matches0[i][0].distance < matches0[i][1].distance * 0.45)   //Selection des matches pertinents
            goodMatches.push_back(matches0[i][0]);
    }

    for(auto& m : goodMatches)  //On ne garde que les poinst cles pertinents
    {
        Point2f p1 = keyPoints1[m.trainIdx].pt;
        Point2f p0 = keyPoints0[m.queryIdx].pt;
        oldkp.push_back(p1);    //sauvegarde des points cles pour la suite
    	kp0H.push_back(p0);     //bons points cles dans l'image 0
    	kp1H.push_back(p1);     //bons points cles dans la frame 1
    }

    Mat H01 = findHomography(kp0H, kp1H, RANSAC);   //Homographie entre l'image 0 et la frame 1 en utilisant les points precedents, et RANSAC.
    Homographies = H01.clone();                     //Sauvegarde de l'homographie pour produit futur
    Mat H1w = H01 * H0w;                            //Matrice d'homographie entre la frame 1 et le monde

    Mat P1w = findPose(H1w);    //Pose frame 1

    vector <Point2f> coordSystem;
    for (const auto& p : coordinateSystem)
    {
        Mat tmp = P1w * p;
        Point2f cs2D = Point2f(tmp.at<double>(0, 0) / tmp.at<double>(2, 0), tmp.at<double>(1, 0) / tmp.at<double>(2, 0));
        coordSystem.push_back(cs2D);
    }
    Mat frame1_axis = frame1.clone();
    line(frame1_axis, coordSystem[0], coordSystem[1], Scalar(0, 0, 255), 2, LINE_AA);
    line(frame1_axis, coordSystem[0], coordSystem[2], Scalar(0, 255, 0), 2, LINE_AA);
    line(frame1_axis, coordSystem[0], coordSystem[3], Scalar(255, 0, 0), 2, LINE_AA);

    vector<KeyPoint> kpDraw1;
    KeyPoint::convert(kp1H, kpDraw1);
    drawKeypoints(frame1_axis, kpDraw1, frame1_axis,Scalar(51,255,255), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    addImg(frame1_axis, img, x, y);

    drawCorners(frame1_axis, corners, imgScale, x, y, H01);

    imshow("Repere et keyPoints", frame1_axis);
    cv::waitKey(22);
    //render.write(frame1_axis);

    //----------------------------------------------------------------------------------
    // Calculs de pose et axes pour la video
    //----------------------------------------------------------------------------------
    {cout <<
        "----------------------------------------------------------------------------------\n Calculs de pose et axes pour la video \n ----------------------------------------------------------------------------------"
        << endl; }

    Mat oldFrame = frame1.clone();
    vector<KeyPoint> keyPoints;
    vector<vector<DMatch>> matches;
    vector<DMatch> goodFlann;
    Mat desc;
    bool f = true;
    auto start = chrono::system_clock::now();
    int i = 0;
    while (f)   //Tant que la video contient des frames
    {
        vector<uchar> status;
        vector<float> err;
        Mat frame;
        f = vid.read(frame);
        if(!f)
        {
            break;
        }
        //resize(frame, frame, Size(frame.cols / 1.5, frame.rows / 1.5));

        Mat Hi;     //Matrice d'homographie entre la frame i-1 et la frame i
        Mat H0i;    //Matrice d'homographie entre l'image 0 et la frame i

        /*
         * Suivi des points cles par KLT. Pour plus de precision, on recalcule un flann toutes les 10 frames.
         */
        if (i < 10) {
            vector<Point2f>KLT, oldKLT;
            calcOpticalFlowPyrLK(oldFrame, frame, oldkp, newkp, status, err, Size(5, 5));

            for (uint i = 0; i < oldkp.size(); i++)     //Selection des points cles pertinents
            {
                if (status[i] == 1) {
                    KLT.push_back(newkp[i]);
                    oldKLT.push_back(oldkp[i]);
                }
            }
        	Hi = findHomography(oldKLT, KLT, RANSAC);   //Homographie
        	H0i = Hi * Homographies;
        	Homographies = H0i.clone();
        	oldkp = KLT;
        }
        else
        {
            goodFlann.clear();
            keyPoints.clear();
            matches.clear();
            sift->detectAndCompute(frame, Mat(), keyPoints, desc);
            matcher->knnMatch(desc0, desc, matches, 2);

            for (unsigned int i = 0; i < matches.size(); ++i) {
                if (matches[i][0].distance < matches[i][1].distance * 0.45)
                    goodFlann.push_back(matches[i][0]);
            }

            if (goodFlann.size() > 10) {        //Si flann a detecte assez de points
                vector<Point2f> oldkpFlann;
                oldkp.clear();
                for (auto& m : goodFlann)
                {
                    Point2f p1 = keyPoints[m.trainIdx].pt;
                    Point2f p0 = keyPoints0[m.queryIdx].pt;
                    oldkp.push_back(p1);
                    oldkpFlann.push_back(p0);
                }
                H0i = findHomography(oldkpFlann, oldkp, RANSAC);
                Homographies = H0i.clone();
            }else                               //Sinon
            {
                vector<Point2f>KLT, oldKLT;
                calcOpticalFlowPyrLK(oldFrame, frame, oldkp, newkp, status, err, Size(5, 5));

                for (uint i = 0; i < oldkp.size(); i++)
                {
                    if (status[i] == 1) {
                        KLT.push_back(newkp[i]);
                        oldKLT.push_back(oldkp[i]);
                    }
                }
                Hi = findHomography(oldKLT, KLT, RANSAC);
                H0i = Hi * Homographies;
                Homographies = H0i.clone();
                oldkp = KLT;
            }
            i = 0;
        }

        Mat Hiw = H0i * H0w;
        Mat Piw = findPose(Hiw);    //Pose a la frame i

        vector<KeyPoint> kpDraw;
        KeyPoint::convert(oldkp, kpDraw);
        coordSystem.clear();
        for (const auto& p : coordinateSystem)
        {
            Mat tmp = Piw * p;
            Point2f cs2D = Point2f(tmp.at<double>(0, 0) / tmp.at<double>(2, 0), tmp.at<double>(1, 0) / tmp.at<double>(2, 0));
            coordSystem.push_back(cs2D);
        }
        Mat frame_axis = frame.clone();
        line(frame_axis, coordSystem[0], coordSystem[1], Scalar(0, 0, 255), 2, LINE_AA);
        line(frame_axis, coordSystem[0], coordSystem[2], Scalar(0, 255, 0), 2, LINE_AA);
        line(frame_axis, coordSystem[0], coordSystem[3], Scalar(255, 0, 0), 2, LINE_AA);

        drawKeypoints(frame_axis, kpDraw, frame_axis,Scalar(51, 255, 255), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        addImg(frame_axis, img, x, y);

        drawCorners(frame_axis, corners, imgScale, x, y, H0i);

        imshow("Repere et keyPoints", frame_axis);
        waitKey(1);

        oldFrame = frame.clone();
        i++;
        //render.write(frame_axis);
    }

	//----------------------------------------------------------------------------------
	// Fin
	//----------------------------------------------------------------------------------
    {cout <<
        "----------------------------------------------------------------------------------\n Fin \n ----------------------------------------------------------------------------------"
        << endl; }

    auto end = chrono::system_clock::now();
    vid.release();
    render.release();
    chrono::duration<double> elapsed_seconds = end - start;
    cout<<"---------------------------------------------------- - "<<endl;
    cout << "elapsed time: " << elapsed_seconds.count() << "s" << endl;
    cout << "---------------------------------------------------- - " << endl;
}



