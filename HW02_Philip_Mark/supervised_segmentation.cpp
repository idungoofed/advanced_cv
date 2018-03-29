#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/***
 * Slightly modified version of https://docs.opencv.org/3.1.0/d2/dbd/tutorial_distance_transform.html
 *
 * Performs segmentation on an image using the watershed algorithm.
 * Steps:
 *     1: Load image
 *     2: Attempt to change the background to black
 *     3: Sharpen the image to emphasize the foreground edges
 *     4: Create a grayscale version and a binarized version of the sharpened image
 *     5: Apply a distance transformation to the binarized image
 *     6: Normalize, threshold, and slightly dilate the image from the previous step
 *     7: Use find contours to delineate the objects, and then use that to create the markers for watershed
 *     8: Pass the markers into the watershed algorithm
 *     9: Visualize the result in a (hopefully) pretty picture
 *
 * Modifications from cited code:
 *     Added thresholding for getting rid of the white background in step 2
 *
 * Trivial Modifications:
 *     Added better, more explanative commenting
 *     Fixed some English
 *     Displays each step as an image (press any key to continue)
 *
 * Notes:
 *     This works best on IMG_7779_shrunk_2ovr.jpg, mostly because of how the thresholding in step 2 is done
 */


int main(int argc, char **argv) {
    // usage message
    string usageMessage = "Usage: supervised_segmentation <image_file>";

    // ensure a parameter is passed
    if (argc != 2) {
        cout << usageMessage << std::endl;
    }

    // Load the image
    Mat src = imread(argv[1]);

    // Check if param is a usable image
    if (!src.data) {
        cout << "Invalid image file." << std::endl;
        cout << usageMessage << std::endl;
        return -1;
    }

    // Show source image
    imshow("Source Image", src);
    waitKey(0);

    // Change the background from white to black, since that will help later
    // with extracting better results during the use of the Distance Transform.
    // Added better thresholding for the white-ish background
    for (int x = 0; x < src.rows; x++) {
        for (int y = 0; y < src.cols; y++) {
            if (src.at<Vec3b>(x, y)[0] >= 200 &&
                src.at<Vec3b>(x, y)[1] >= 200 &&
                src.at<Vec3b>(x, y)[2] >= 200) {
                src.at<Vec3b>(x, y) = Vec3b(0, 0, 0);
            }
        }
    }

    // Show output image
    imshow("Black Background Image", src);
    waitKey(0);


    // Create a kernel (approximation of 2nd laplacian derivative) that we will use for sharpening our image
    Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
    Mat imgLaplacian;
    Mat sharp;
    src.copyTo(sharp); // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bit grayscale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    imgResult.copyTo(src); // copy back
    imshow("Sharpened Image", src);
    waitKey(0);


    // Create binary image from source image using otsu binarization
    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
    threshold(bw, bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    imshow("Binarized Image", bw);
    waitKey(0);

    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0} so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    waitKey(0);

    // Threshold to obtain the peaks
    // These will be the markers for the foreground objects
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);

    // Dilate the dist image a bit to make the foreground objects more solid
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(dist, dist, kernel1);
    imshow("Peaks", dist);
    waitKey(0);

    // Create a CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Find markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);

    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i) + 1), -1);

    // Draw the background marker and display
    circle(markers, Point(5, 5), 3, CV_RGB(255, 255, 255), -1);
    imshow("Markers", markers * 10000);
    waitKey(0);

    // Perform the watershed algorithm
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);

    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++) {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar) b, (uchar) g, (uchar) r));
    }

    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);

    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst.at<Vec3b>(i, j) = colors[index - 1];
            else
                dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
        }
    }

    // Visualize the final image
    imshow("Final Result", dst);
    waitKey(0);
    return 0;
}