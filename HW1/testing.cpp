#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    // open the images
    Mat diff1 = imread("diff1.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat diff2 = imread("diff2.png", CV_LOAD_IMAGE_GRAYSCALE);

    /*
    // split it into its 3 color planes
    int cols, rows;
    cols = diff1.cols;
    rows = diff1.rows;

    // make sure diff1 and diff2 have same rows/cols
    assert(cols == diff2.cols && rows == diff2.rows);
     */
    // create output holder
    cv::Mat diff;
    cv::compare(diff1, diff2, diff, cv::CMP_NE);
    int nz = cv::countNonZero(diff);
    cout << nz << std::endl;


    namedWindow("output", WINDOW_GUI_NORMAL);
    resizeWindow("output", 600, 600);
    imshow("output", diff);
    waitKey(0);
}