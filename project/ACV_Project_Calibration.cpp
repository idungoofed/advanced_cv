//
// Created by Ben Brown on 4/5/18.
//

/**
 * Ben Brown : beb5277
 * Mark Philip : msp3430
 * Advanced Computer Vision Project
 * Reads in a warped image with crosses and performs a Hough transform to locate
 * the corners of the image
 *
 * Ellipse detection: https://www.sciencedirect.com/science/article/pii/S0031320314001976
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv/cv.hpp>


#include <iostream>

using namespace cv;
using namespace std;


const int minCircleArea = 10;

// edge circle colors
const Scalar tl_color(0,0,255); // hsv(0, 255, 255)
const Scalar tr_color(0,255,127); // hsv(45, 255, 255)
const Scalar bl_color(255,255,0); // hsv(90, 255, 255)
const Scalar br_color(255,0,127); // hsv(135, 255, 255)

/*
 * Gets average rgb value of the given mat (blob) and returns the hsv value
 */
Scalar bgrToHSV(Mat bgr) {
    imshow("pt", bgr);
    waitKey(0);
    destroyWindow("pt");
    Scalar avg = mean(bgr); //bgr
    Mat bgr_one(1,1, CV_8UC3, avg);
    cout << "AVG: " << avg << endl;
    cout << "BGR: " << bgr_one << endl;
    Mat hsv_one;
    cvtColor(bgr_one, hsv_one, CV_BGR2HSV);
    Scalar hsv = hsv_one.at<Vec3b>(Point(0, 0));
    cout << "HSV: " << hsv << endl;
    return hsv;
}

//Returns corners in order: TL, TR, BR, BL
vector<KeyPoint> getCorners(Mat image) {
    /* FLOW
     * init: four arrays: one for each corner
     * detect blobs
     * for each blob:
     *    convert to hsv
     *    if color is close to one of the presets:
     *       if in the same quadrant:
     *          append to correct array
     * find most extreme blob in each array -> corner_points
    */

    Mat grayscaleMat;
    cvtColor(image, grayscaleMat,CV_BGR2GRAY);

    SimpleBlobDetector::Params params;
    params.filterByColor = false;
    params.filterByArea = true;
    params.minArea = minCircleArea;
    params.filterByCircularity = true;
    params.minCircularity = .8;
    params.filterByConvexity = true;
    params.minConvexity = .9;
    params.filterByInertia = true;
    params.minInertiaRatio = .5;


    vector<KeyPoint> tl_list, tr_list, bl_list, br_list;



    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    vector<KeyPoint> keypoints;
    detector->detect(grayscaleMat, keypoints);
    for (const auto& item : keypoints) {
        cout << item.pt << ": " << item.size << endl;

        Rect extractedBlob((int)round(item.pt.x - item.size/4),
                           (int)round(item.pt.y - item.size/4),
                           (int)round(item.size/2), (int)round(item.size/2));
        Mat forConversion = Mat(image, extractedBlob);
        Scalar hsv = bgrToHSV(forConversion);
        /*
         * if color is close to one of the presets:
         *    if in the same quadrant:
         *       append to correct array
         */

        cout << endl;
    }

    return keypoints;
}


int main(int argc, char** argv) {

    Mat testImage(Size(400, 400), CV_8UC3, Scalar(255, 255, 255));

    int edgeMargin = 25;
    int radius = 20;

    Scalar crossColor = Scalar(0, 0, 255);

    //Create corners (centers of crosses)
    Point topLeft(edgeMargin, edgeMargin);
    Point topRight(testImage.rows - edgeMargin, edgeMargin);
    Point bottomRight(testImage.rows - edgeMargin, testImage.cols - edgeMargin);
    Point bottomLeft(edgeMargin, testImage.cols - edgeMargin);



    vector<Point2f> points; //Clockwise: TL, TR, BR, BL

    cout << "Points:" << endl;
    cout << "\tTL: " << topLeft << ", " << tl_color << endl;
    cout << "\tTR: " << topRight << ", " << tr_color << endl;
    cout << "\tBL: " << bottomLeft << ", " << bl_color << endl;
    cout << "\tBR: " << bottomRight << ", " << br_color << endl << endl;

    // draw colored circles on the points
    circle(testImage, topLeft, radius, tl_color, -1);
    circle(testImage, topRight, radius, tr_color, -1);
    circle(testImage, bottomLeft, radius, bl_color, -1);
    circle(testImage, bottomRight, radius, br_color, -1);

    // display the original image
    imshow("Original Image", testImage);

    // find the blobs
    vector<KeyPoint> keypoints = getCorners(testImage);
    for (const auto& kp : keypoints) {
        circle(testImage, kp.pt, int(kp.size)/4, Scalar(255, 255, 255), -1);
    }
    // display
    imshow("Found Circles", testImage);
    waitKey(0);

    // code for capturing a pic
    /*
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening webcam" << endl;
        return EXIT_FAILURE;
    }
    Mat webcam;
    const char *webcam_window = "Webcam";
    namedWindow(webcam_window, CV_WINDOW_NORMAL);
    setWindowProperty(webcam_window, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    for (;;) {
        Mat frame;
        cap >> frame;
        imshow(webcam_window, frame);
        if (waitKey(30) >= 0) {
            imwrite("./webcam_output.png", frame);
            break;
        }
    }
    */
    return EXIT_SUCCESS;
}