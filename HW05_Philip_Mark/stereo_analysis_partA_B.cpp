/** Author: Mark Philip (msp3430)
 *
 * Inspired by:
 *     - https://github.com/sourishg/stereo-calibration/blob/master/calib_intrinsic.cpp
 *     - https://github.com/daviddoria/Examples/blob/master/c%2B%2B/OpenCV/CheckerBoardCalibration/CalibrateAndDisplay.cxx
 *     - https://docs.opencv.org/3.1.0/d4/d94/tutorial_camera_calibration.html
 */


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
#include <dirent.h>

using namespace cv;
using namespace std;

/** Transforms 2-D coordinates of a point into hypothetical 3-D points.
 *
 * @param boardSize     The size of the board. In this case, 8x6
 * @param squareSize    The size of a square on the board
 * @return              A vector of 3-D points
 */
std::vector<cv::Point3f> Create3DChessboardCorners(cv::Size boardSize, float squareSize) {
    std::vector<cv::Point3f> corners;
    // map the 2-D coords to 3-D coords
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            corners.push_back(
                    cv::Point3f(
                            float(j * squareSize),
                            float(i * squareSize),
                            0
                    )
            );
        }
    }
    return corners;
}

/** Takes an image filename and calculates the camera matrix and distortion coefficients.
 *
 * @param imageFileName     The path to the image to process
 * @return                  True if the image was processed successfully, else false
 */
bool processImage(char *imageFileName) {
    // size of the board
    cv::Size boardSize(8, 6);
    float squareSize = 1.f; // This is "1 arbitrary unit"

    // open the image
    cv::Mat raw_image = cv::imread(imageFileName);
    if (raw_image.empty()) {
        std::cerr << "Image not read correctly!" << std::endl;
        return false;
    }

    // downsize image by a factor of 4
    Mat image;
    resize(raw_image, image, raw_image.size() / 4);

    // retrieve once for multiple uses
    cv::Size imageSize = image.size();

    // array of 2-D image points to be mapped to 3-D points later
    vector<std::vector<cv::Point2f> > imagePoints(1);

    // Find the chessboard corners
    Mat imageGray;
    // create a grayscale version for use with cornerSubPix()
    cvtColor(image, imageGray, CV_BGR2GRAY);
    // get the corners using adaptive thresholding
    bool found = findChessboardCorners(image, boardSize, imagePoints[0],
                                       CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
    if (found) {
        // if the corners were found, refine the point locations
        cornerSubPix(imageGray, cv::Mat(imagePoints[0]), cv::Size(5, 5), cv::Size(-1, -1),
                     TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
        // draw the points on the color image
        drawChessboardCorners(image, boardSize, cv::Mat(imagePoints[0]), found);
        // create and fill the array of 3-D image points
        std::vector<std::vector<cv::Point3f> > objectPoints(1);
        objectPoints[0] = Create3DChessboardCorners(boardSize, squareSize);

        // prep params for calibration
        std::vector<cv::Mat> rotationVectors;
        std::vector<cv::Mat> translationVectors;
        cv::Mat distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F); // There are 8 distortion coefficients
        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

        // calibrate!
        int flags = 0 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5;
        double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                                     distortionCoefficients, rotationVectors, translationVectors,
                                     flags);

        // print the params
        std::cout << "RMS: " << rms << std::endl;

        std::cout << "Camera matrix: " << cameraMatrix << std::endl;
        std::cout << "Distortion coefficients: " << distortionCoefficients << std::endl;

        // Unmap the image to remove the distortion
        Mat undistorted, map1, map2;

        // create maps and remap
        initUndistortRectifyMap(
                cameraMatrix, distortionCoefficients, Mat(),
                getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients,
                                          imageSize, 1, imageSize),
                imageSize, CV_16SC2, map1, map2
        );
        remap(image, undistorted, map1, map2, INTER_LINEAR);

        Mat imageDiff;
        absdiff(image, undistorted, imageDiff);

        // show the original and undistorted images, and also the difference between them
        namedWindow("Original Image", WINDOW_GUI_NORMAL);
        namedWindow("Undistorted Image", WINDOW_GUI_NORMAL);
        namedWindow("Differences", WINDOW_GUI_NORMAL);
        imshow("Original Image", image);
        imshow("Undistorted Image", undistorted);
        imshow("Differences", imageDiff);
        waitKey(0);
    }
    return found;
}

int main(int argc, char *argv[]) {
    // get all the files in ./IMAGES_02_CALIBRATION/
    DIR *dir;
    struct dirent *ent;
    bool done = false;
    // keep processing images until one of them succeeds
    if ((dir = opendir("./IMAGES_02_CALIBRATION/"))) {
        while (!done && (ent = readdir(dir))) {
            // exclude . and ..
            if (strcmp(ent->d_name, ".\0") && strcmp(ent->d_name, "..\0")) {
                // create the filename string
                char *filename = (char *) malloc(sizeof(char) * (25 + strlen(ent->d_name)));
                filename = strncpy(filename, "./IMAGES_02_CALIBRATION/\0", 25 + strlen(ent->d_name));
                filename = strcat(filename, ent->d_name);
                // process the image
                printf("Now processing: \"%s\"\n", filename);
                done = processImage(filename);

                // free the earlier-malloced filename
                free(filename);
            }
        }
        closedir(dir);
    } else {
        // could not open directory
        std::cout << "Error! ./IMAGES_02_CALIBRATION/ doesn't exist." << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}