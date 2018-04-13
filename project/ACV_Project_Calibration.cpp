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

// edge circle colors
const Scalar tl_color(0,0,255); // hsv(0, 255, 255)
const Scalar tl_hsv(0,255,255);

const Scalar tr_color(0,255,127); // hsv(45, 255, 255)
const Scalar tr_hsv(45,255,255);

const Scalar bl_color(255,255,0); // hsv(90, 255, 255)
const Scalar bl_hsv(90,255,255);

const Scalar br_color(255,0,127); // hsv(135, 255, 255)
const Scalar br_hsv(135,255,255);

// acceptable error
const int hue_error = 15;
const int saturation_error = 10;

// constants for readability
const int H_VAL = 0;
const int S_VAL = 1;

/*
 * Gets average rgb value of the given mat (extracted blob) and returns its hsv value
 */
Scalar bgrToHSV(Mat bgr) {
    //imshow("pt", bgr);
    //waitKey(0);
    //destroyWindow("pt");
    Scalar avg = mean(bgr);
    Mat bgr_one(1,1, CV_8UC3, avg);
    cout << "AVG: " << avg << endl;
    cout << "BGR: " << bgr_one << endl;
    Mat hsv_one;
    cvtColor(bgr_one, hsv_one, CV_BGR2HSV);
    Scalar hsv = hsv_one.at<Vec3b>(Point(0, 0));
    cout << "HSV: " << hsv << endl;
    return hsv;
}

int getQuadrant(Point pt, Size imgSize) {
    Point mdpt(imgSize.width/2, imgSize.height/2);
    bool pt_left = mdpt.x - pt.x > 0;
    bool pt_up = mdpt.y - pt.y > 0;
    return pt_left ? (pt_up ? 2 : 3) : (pt_up ? 1 : 4);
}

bool checkColor(Scalar candidate, Scalar original) {
    //return true;
    double upper_bound = original[H_VAL] + hue_error;
    double lower_bound = original[H_VAL] - hue_error;
    if (lower_bound < 0) {
        lower_bound+= 180;
        if (lower_bound < candidate[H_VAL] || candidate[H_VAL] < upper_bound) {
            return true;
        }
        else {
            return false;
        }
    }
    else {
        if (lower_bound < candidate[H_VAL] < upper_bound) {
            return true;
        }
        else {
            return false;
        }
    }
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

    /* flow:
     *
     *
     *
     *
     */



    vector<KeyPoint> keypoints, tl_list, tr_list, bl_list, br_list;
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    detector->detect(grayscaleMat, keypoints);

    for (const auto& item : keypoints) {
        cout << item.pt << ": " << item.size << endl;

        Rect extractedBlob((int)round(item.pt.x - item.size/4),
                           (int)round(item.pt.y - item.size/4),
                           (int)round(item.size/2), (int)round(item.size/2));
        Mat forConversion = Mat(image, extractedBlob);
        Scalar hsv = bgrToHSV(forConversion);
        int quadrant = getQuadrant(item.pt, image.size());
        cout << "Quadrant: " << quadrant << endl;
        switch (quadrant) {
            case 1:
                if (checkColor(hsv, tr_hsv)) {
                    cout << "Added " << item.pt << " to TR" << endl;
                    tr_list.push_back(item);
                }
                break;
            case 2:
                if (checkColor(hsv, tl_hsv)) {
                    cout << "Added " << item.pt << " to TL" << endl;
                    tl_list.push_back(item);
                }
                break;
            case 3:
                if (checkColor(hsv, bl_hsv)) {
                    cout << "Added " << item.pt << " to BL" << endl;
                    bl_list.push_back(item);
                }
                break;
            case 4:
                if (checkColor(hsv, br_hsv)) {
                    cout << "Added " << item.pt << " to BR" << endl;
                    br_list.push_back(item);
                }
                break;
            default:
                cout << "Error: Quadrant error" << endl;
                break;
        }
        cout << endl;
    }
    // ToDo: find corner-est of each list, remove all others


    if (not (tl_list.empty() || tr_list.empty() || br_list.empty() || bl_list.empty()) ) {
        return vector<KeyPoint>{tl_list[0], tr_list[0], br_list[0], bl_list[0]};
    }
    else {
        return vector<KeyPoint>{};
    }
}


int main(int argc, char** argv) {

    /*
    Mat testImage(Size(400, 400), CV_8UC3, Scalar(255, 255, 255));

    int edgeMargin = 25;
    int radius = 20;

    Scalar crossColor = Scalar(0, 0, 255);

    //Create corners (centers of circles)
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
    */

    Mat testImage = imread("./test_pics/webcam_output.png");

    // display the original image
    imshow("Original Image", testImage);

    // find the blobs
    vector<KeyPoint> keypoints = getCorners(testImage);
    cout << "Found " << keypoints.size() << " keypoints" << endl;
    for (const auto& kp : keypoints) {
        circle(testImage, kp.pt, int(kp.size)/4, Scalar(255, 255, 255), -1);
    }
    // display
    imshow("Found Circles", testImage);
    waitKey(0);

    // code for capturing a pic
    /*
    destroyAllWindows();
    namedWindow("x", CV_WINDOW_NORMAL);
    imshow("x", testImage);
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
        if ((char)waitKey(30) == 'q') {
            imwrite("./webcam_output.png", frame);
            break;
        }
    }
    */

    return EXIT_SUCCESS;
}