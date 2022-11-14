#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

	Mat I0full = imread(R"(C:\Users\user\Documents\_ESIRTP\3\VROB\imtest.jpg)");
	Mat I0;

	resize(I0full, I0, Size(I0.cols / 2, I0.rows / 2)); //width, height


}



