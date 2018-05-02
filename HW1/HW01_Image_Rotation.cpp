#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv/cv.hpp>

using namespace cv;
using namespace std;


/* INSTRUCTIONS
 * ------------
 * Create a C++/OpenCV program to load a *.jpg image and display it rotated.
 * Find a small cat image from somewhere and display it rotated counter-clock-wise by 90 degrees.
 * This will involve you figuring out how to rotate an image in OpenCV.
 * The book doesn’t bother covering this you will have to search docs.opencv.org or the WWW.
 * You program should successfully handle the case where it is asked to open a file that does not exist.
 * In this case it should report an error and exit gracefully (with an error code).
 * Using a bogus file should not cause a segmentation fault.
 * Fully comment your code.
 * Write up how you did this in your write-up.
 */

Mat rotateImage(Mat src) {
    // get the center of the image
    cv::Point2f center(src.cols/2.0F, src.rows/2.0F);
    // create a transformation matrix: rotate 90 around the center with no scaling
    Mat trfm = cv::getRotationMatrix2D(center, 90, 1);
    // destination for transformation
    Mat dest;
    // do the transformation
    cv::warpAffine(src, dest, trfm, src.size());
    return dest;
}

int main(int argc, char *argv[]) {
    string usageMessage = "Usage: HW01_Image_Rotation <image.jpg>";
    // ensure a parameter for the image file was provided
    if (argc != 2) {
        std::cout << "Error: Missing image file!" << std::endl << usageMessage << std::endl;
        return EXIT_FAILURE;
    }
    // ensure that the given image filename is a .jpg
    int arglen = (int)strlen(argv[1]);
    if (arglen < 4 ||
            argv[1][arglen-4] != '.' ||
            argv[1][arglen-3] != 'j' ||
            argv[1][arglen-2] != 'p' ||
            argv[1][arglen-1] != 'g') {
        std::cout << "Error: File is not a jpg!" << std::endl << usageMessage << std::endl;
        return EXIT_FAILURE;
    }

    // attempt to load the image
    Mat image = imread(argv[1], CV_LOAD_IMAGE_ANYCOLOR);

    // if image was valid
    if (image.data) {
        // rotate the image
        Mat dest = rotateImage(image);
        // display the image
        String window_title = "Rotated image";
        namedWindow(window_title, CV_WINDOW_NORMAL);
        resizeWindow(window_title, 800, 800);
        imshow(window_title, dest);
        // wait for a keypress to exit
        waitKey(0);
        return EXIT_SUCCESS;
    }
    // invalid image file
    else {
        cout << "Error: Invalid image file!" << std::endl << usageMessage << std::endl;
        return EXIT_FAILURE;
    }
}
