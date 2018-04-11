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
//#include <opencv2/videoio/videoio.hpp>
//#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

//#include <map>
#include <iostream>
//#include <fstream>
#include <opencv/cv.hpp>

using namespace cv;
using namespace std;

struct LineStruct {
    Point2f startpt;
    Point2f endpt;
};

//Mat firstFrame;
//String window = "Image Display";
//int lineError = 5;
int binarizeThreshold = 70;
//int pointDistError = 3;
const int minCircleArea = 10;

//Returns corners in order: TL, TR, BR, BL
vector<KeyPoint> getCornersV2(Mat image) {
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

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    std::vector<KeyPoint> keypoints;
    detector->detect(grayscaleMat, keypoints);
    for (const auto& item : keypoints) {
        cout << item.pt << ": " << item.size << endl;

        /*
         * convert to hsv
         * if color is close to one of the presets:
         * if in the same quadrant:
         *    append to correct array
         */
    }

    return keypoints;



    /*
// Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;

// Filter by Area.
params.filterByArea = true;
params.minArea = 1500;

// Filter by Circularity
params.filterByCircularity = true;
params.minCircularity = 0.1;

// Filter by Convexity
params.filterByConvexity = true;
params.minConvexity = 0.87;

// Filter by Inertia
params.filterByInertia = true;
params.minInertiaRatio = 0.01;
     */
}

//Returns corners in order: TL, TR, BR, BL
vector<Point2f> getCorners(Mat image) {
    vector<Point2f> crossLocations;
    vector<LineStruct> horizLines;
    vector<LineStruct> vertLines;


    Mat grayscaleMat;
    cvtColor(image, grayscaleMat, COLOR_BGR2GRAY);

    /*
    // first find edges
    Mat cannyEdges;
    Canny(grayscaleMat, cannyEdges, 10, 50, 3);
    imshow("Canny Edges", cannyEdges);
    */

    // binarize
    Mat binarizedImg;
    threshold(grayscaleMat, binarizedImg, binarizeThreshold, 255, CV_THRESH_BINARY);
    imshow("Binarized", binarizedImg);

    // now use edges to find lines
    vector<Vec4i> houghLines;
    HoughLinesP(binarizedImg, houghLines, 1, CV_PI/180, 30, 5, 2);
    cout << "Lines found: " << houghLines.size() << endl;

    // mat for displaying found lines
    Mat foundLines = Mat(image.rows, image.cols, CV_8UC3, Scalar(0, 0, 0));

    // hashmap for keeping track of found points

    for (const auto& foundLine : houghLines) {
        Point pt1 = Point(foundLine[0], foundLine[1]);
        Point pt2 = Point(foundLine[2], foundLine[3]);
        line(foundLines, pt1, pt2, Scalar(0, 0, 255));
        cout << "Pt1: " << pt1 << ", Pt2: " << pt2 << endl;
    }
    /*
    for (size_t idx = 0; idx < houghLines.size(); idx++) {
        Point pt1 = Point(houghLines[idx][0], houghLines[idx][1]);
        Point pt2 = Point(houghLines[idx][2], houghLines[idx][3]);
        line(foundLines, pt1, pt2, Scalar(0, 0, 255));
        cout << "Pt1: " << pt1 << ", Pt2: " << pt2 << endl;
    }
    */
    imshow("Found Lines", foundLines);

    // TODO: cluster points with a distance of < pointDistError away

    /*
    // original attempt with crosses
    vector<Vec2f> houghLines;
    HoughLines(cannyEdges, houghLines, 1, CV_PI/180, 30, 0, 0);

    cout << houghLines.size() << "\n";

    for( size_t i = 0; i < houghLines.size(); i++ )
    {
        float rho = houghLines[i][0], theta = houghLines[i][1];
        cout << theta << endl;
        // find vertical lines
        if (0-lineError < int(theta)%180 < 0+lineError) {// || 85 < theta < 95) {
            cout << "found vert line" << endl;
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            LineStruct ls;
            ls.startpt = Point(
                    cvRound(x0 + 1000*(-b)),
                    cvRound(y0 + 1000*(a))
            );
            ls.endpt = Point(
                    cvRound(x0 - 1000*(-b)),
                    cvRound(y0 - 1000*(a))
            );
            vertLines.push_back(ls);
        }
        //find horizontal lines
        else if (90-lineError < theta < 90+lineError) {
            cout << "found horiz line" << endl;
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            LineStruct ls;
            ls.startpt = Point(
                    cvRound(x0 + 1000*(-b)),
                    cvRound(y0 + 1000*(a))
            );
            ls.endpt = Point(
                    cvRound(x0 - 1000*(-b)),
                    cvRound(y0 - 1000*(a))
            );
            cout << ls.startpt << endl;
            cout << ls.endpt << endl;
            horizLines.push_back(ls);
        }
        // ignore all other lines
    }

    // draw the found lines just for visualization
    for (size_t idx = 0; idx < vertLines.size(); idx++) {
        line(image, vertLines[idx].startpt, vertLines[idx].endpt, Scalar(0, 0, 255));
    }
    for (size_t idx = 0; idx < horizLines.size(); idx++) {
        line(image, horizLines[idx].startpt, horizLines[idx].endpt, Scalar(0, 0, 255));
    }
    imshow("Found Lines", image);
    // find intersections between lines



    //imshow("Found Corners", image);

     */
    return crossLocations;
}

int main(int argc, char** argv) {

    Mat testImage(Size(400, 400), CV_8UC3);

    int edgeMargin = 25;
    int crossWidth = 20;
    int crossThickness = 2;

    Scalar crossColor = Scalar(0, 0, 255);

    //Create corners (centers of crosses)
    Point topLeft(edgeMargin, edgeMargin);
    Point topRight(testImage.rows - edgeMargin, edgeMargin);
    Point bottomRight(testImage.rows - edgeMargin, testImage.cols - edgeMargin);
    Point bottomLeft(edgeMargin, testImage.cols - edgeMargin);

    vector<Point2f> points; //Clockwise: TL, TR, BR, BL

    cout << "Points:" << endl;
    cout << "\tTL: " << topLeft << endl;
    cout << "\tTR: " << topRight << endl;
    cout << "\tBL: " << bottomLeft << endl;
    cout << "\tBR: " << bottomRight << endl;

    // draw colored circles on the points
    circle(testImage, topLeft, crossWidth, Scalar(0,0,255), -1); // hsv(0, 100, 100)
    circle(testImage, topRight, crossWidth, Scalar(0,255,127), -1); // hsv(90, 100, 100)
    circle(testImage, bottomLeft, crossWidth, Scalar(255,255,0), -1); // hsv(180, 100, 100)
    circle(testImage, bottomRight, crossWidth, Scalar(255,0,127), -1); // hsv(270, 100, 100)

    // display the original image
    imshow("Original Image", testImage);
    destroyAllWindows();
    vector<KeyPoint> keypoints = getCornersV2(testImage);
    for (const auto& kp : keypoints) {
        circle(testImage, kp.pt, int(kp.size)/4, Scalar(255, 255, 255), -1);
    }
    imshow("Found Circles", testImage);
    waitKey(0);
    /*
    // draw borders
    line(testImage, topLeft, topRight, crossColor, crossThickness);
    line(testImage, topLeft, bottomLeft, crossColor, crossThickness);
    line(testImage, topRight, bottomRight, crossColor, crossThickness);
    line(testImage, bottomLeft, bottomRight, crossColor, crossThickness);
     */

    /*
    points.push_back(topLeft);
    points.push_back(topRight);
    points.push_back(bottomRight);
    points.push_back(bottomLeft);

    for (int point=0; point<4; point++) {
        Point thePoint = points[point];

        line(testImage,
             Point(thePoint.x - crossWidth, thePoint.y),
             Point(thePoint.x + crossWidth, thePoint.y),
             crossColor, crossThickness
        );
        line(testImage,
             Point(thePoint.x, thePoint.y - crossWidth),
             Point(thePoint.x, thePoint.y + crossWidth),
             crossColor, crossThickness
        );
    }
    */
    /*
    // display the original image
    imshow("Original Image", testImage);
    waitKey(0);
    // find the (red) corners

    vector<Point2f> calculatedCorners = getCorners(testImage);

    // draw circles around the corners that were detected

    for (int point=0; point<calculatedCorners.size(); point++) {
        circle(testImage, calculatedCorners[point], 5, Scalar(0,255,0), -1);
    }
    imshow("Found Corners", testImage);
    */



    /*
    waitKey(0);
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