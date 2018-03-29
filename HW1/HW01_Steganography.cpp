#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

/*
 * Reading each bit plane inspired by https://stackoverflow.com/a/36410072/5496345
 */
int main(int argc, char *argv[]) {
    // open the image
    Mat image = imread("CAT_Kitten_img_13.jpg", CV_LOAD_IMAGE_ANYCOLOR);
    // split it into its 3 color planes
    Mat bgr[3];
    split(image, bgr);
    int cols, rows;
    cols = image.cols;
    rows = image.rows;
    // iterate over each color plane
    for(int colr = 0; colr < 3; colr++) {
        // iterate over each bit plane
        for (int bitplane = 1; bitplane < 129; bitplane*= 2) {
            // create a binarized version of this bitplane
            Mat output(rows, cols, CV_8UC1, Scalar(0));
            for (int y = 0; y < rows; y++) {
                for (int x = 0; x < cols; x++) {
                    // bitwise-and this pixel with bitplane to get value of this pixel at this bitplane
                    output.at<uchar>(y, x) = (bgr[colr].at<uchar>(y, x) & uchar(bitplane)) ? uchar(255) : uchar(0);
                }
            }
            // display the output image
            namedWindow("output", WINDOW_GUI_NORMAL);
            resizeWindow("output", 800, 800);
            imshow("output", output);
            waitKey(0);
        }
    }
}