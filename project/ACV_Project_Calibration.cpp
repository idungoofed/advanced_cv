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
#include <opencv/cv.hpp>


#include <iostream>

using namespace cv;
using namespace std;

// drawing constants
const int calibration_circle_radius = 20;
int pic_diff = 3;

// constants for readability
const int H_VAL = 0;
const int S_VAL = 1;

// window names
const char *calibration_window_name = "Calibration Window";
const char *webcam_window = "Webcam";
const char *difference_window = "Calibration";

/*
 * Gets average rgb value of the given mat (extracted blob) and returns its hsv value
 */
Scalar BGRtoHSV(Mat bgr) {
    //imshow("pt", bgr);
    //waitKey(0);
    //destroyWindow("pt");
    Scalar avg = mean(bgr);
    Mat bgr_one(1,1, CV_8UC3, avg);
    //cout << "AVG: " << avg << endl;
    //cout << "BGR: " << bgr_one << endl;
    Mat hsv_one;
    cvtColor(bgr_one, hsv_one, CV_BGR2HSV);
    Scalar hsv = hsv_one.at<Vec3b>(Point(0, 0));
    cout << "HSV: " << hsv << endl;
    return hsv;
}

Scalar HSVtoBGR(Scalar hsv) {
    Mat hsv_one(1, 1, CV_8UC3, hsv);
    Mat bgr_one;
    cvtColor(hsv_one, bgr_one, CV_HSV2BGR);
    Scalar bgr = bgr_one.at<Vec3b>(Point(0, 0));
    return bgr;
}

void drawCircles(vector<Point2f> circles, int hue_val, Mat image) {
    for (const auto& circ : circles) {
        circle(image, circ, calibration_circle_radius, HSVtoBGR(Scalar(hue_val, 255, 255)), -1);
    }
}

vector<KeyPoint> getDifferences(const Mat prevFrame, const Mat currFrame) {
    Mat diff = Mat(currFrame.rows, currFrame.cols, currFrame.type());
    absdiff(currFrame, prevFrame, diff);
    cvtColor(diff, diff, CV_BGR2GRAY);
    diff = diff > 5;//pic_diff;
    /*
    if (countNonZero(diff) > (diff.rows * diff.cols)/8) {
        pic_diff++;
        cout << "Difference threshold: " << pic_diff << endl;
    }
     */
    //else if (pic_diff > 1){
    //    pic_diff--;
    //}

    imshow(difference_window, diff);

    return vector<KeyPoint>{};
}

//Returns corners in order: TL, TR, BR, BL
vector<KeyPoint> getCorners() {
    // create the calibration image and display it
    Mat testImage(Size(400, 400), CV_8UC3, Scalar(255, 255, 255));

    int edgeMargin = 25;

    Scalar crossColor = Scalar(0, 0, 255);

    //Create corners (centers of circles)
    Point topLeft(edgeMargin, edgeMargin);
    Point topRight(testImage.rows - edgeMargin, edgeMargin);
    Point bottomRight(testImage.rows - edgeMargin, testImage.cols - edgeMargin);
    Point bottomLeft(edgeMargin, testImage.cols - edgeMargin);

    vector<Point2f> points; //Clockwise: TL, TR, BR, BL
    points.push_back(topLeft);
    points.push_back(topRight);
    points.push_back(bottomRight);
    points.push_back(bottomLeft);
    cout << "Points:" << endl;
    cout << "\tTL: " << topLeft << ", " << endl;
    cout << "\tTR: " << topRight << ", " << endl;
    cout << "\tBL: " << bottomLeft << ", " << endl;
    cout << "\tBR: " << bottomRight << ", " << endl;

    // display the calibration image
    namedWindow(calibration_window_name, CV_WINDOW_NORMAL);
    imshow(calibration_window_name, testImage);

    // setup display for difference display
    namedWindow(difference_window, CV_WINDOW_NORMAL);

    // create the webcam capture and prep for displaying it
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening webcam" << endl;
        return vector<KeyPoint>{};
    }
    namedWindow(webcam_window, CV_WINDOW_NORMAL);
    setWindowProperty(webcam_window, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

    // grab the first two images
    Mat prevFrame;
    Mat currFrame;
    drawCircles(points, 0, testImage);
    cap >> prevFrame;
    //prevFrame = testImage.clone();
    drawCircles(points, 3, testImage);
    cap >> currFrame;
    //currFrame = testImage.clone();

    vector<KeyPoint> num_points = getDifferences(prevFrame, currFrame); // displays overall binary difference on differences_window
    while (num_points.size() != 4) {
        for (int i = 9; i < 180; i += 3) {
            drawCircles(points, i, testImage);
            prevFrame = currFrame.clone();
            cap >> currFrame;
            //currFrame = testImage.clone();
            imshow(calibration_window_name, testImage);
            imshow(webcam_window, currFrame);
            waitKey(30);
        }
        num_points = getDifferences(prevFrame, currFrame); // displays overall binary difference on differences_window
    }
    return num_points;

}


int main(int argc, char** argv) {
    getCorners();
    return EXIT_SUCCESS;
}