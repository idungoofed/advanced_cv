/**
 * Ben Brown : beb5277
 * Mark Philip : msp3430
 * Advanced Computer Vision Project
 *
 * Maps a webcam image of a projection back onto the original image that is being projected, allowing for a user with
 * a laser pointer to interact with the laptop just by pointing the laser at the projector screen.
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <map>
#include <iostream>
#include <fstream>


// #include <opencv2/opencv.hpp>


using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// drawing constants
const int calibration_circle_radius = 20;
const Size window_size = Size(1024,768);

// filter params
int pic_diff = 3;
int num_cap_frames = 18;
int curr_num_dilations = 3;

// window names
const char *calibration_window = "Calibration Window";
const char *webcam_window = "Webcam Image";
const char *difference_window = "Calibration";
const char *found_points_window = "Found Points";
const char *rect_roi_image = "Rectified ROI";

// Ben
Mat firstFrame;
Mat board;
Point lastPointOnBoard;
const char *window = "Image Display";

//https://docs.opencv.org/3.1.0/d5/d6f/tutorial_feature_flann_matcher.html
bool getLocationOfLaserPoint(Mat rectifiedPresentationView, Mat currentlyDisplayed, Point laserPointOut) {

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create();
    detector->setHessianThreshold(minHessian);

    std::vector<KeyPoint> kpCamera, kpKnown;
    Mat descriptorsCamera, descriptorsKnown;
    detector->detectAndCompute(
            rectifiedPresentationView, Mat(), kpCamera, descriptorsCamera
    );
    detector->detectAndCompute(
            currentlyDisplayed, Mat(), kpKnown, descriptorsKnown
    );

    //-- Step 2: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptorsCamera, descriptorsKnown, matches);

    double maxDist = 0, minDist = 100;
    // Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptorsCamera.rows; i++ )  {
        double dist = matches[i].distance;
        if( dist < minDist ) minDist = dist;
        if( dist > maxDist ) maxDist = dist;
    }

    //-- Step 3: Get average distance, X and Y

    double averageXDistSum = 0.0, averageYDistSum = 0.0;
    int numberOfGoodMatches = 0;
    double averageXDist = 0, averageYDist = 0;

    vector<Point2f> queryPts;
    vector<Point2f> matchedPts;

    for( int i = 0; i < descriptorsCamera.rows; i++ ) {
        if( matches[i].distance <= max(2.0*minDist, 0.02) ) {
            //It's a good match! Adjust average x and y dist
            /*Point2f pointCamera = kpCamera[matches[i].queryIdx].pt;
            Point2f pointKnown = kpCamera[matches[i].trainIdx].pt;

            averageXDistSum += pointKnown.x - pointCamera.x;
            averageYDistSum += pointKnown.y - pointCamera.y;
            numberOfGoodMatches ++;*/

            queryPts.push_back( kpCamera[matches[i].queryIdx].pt );
            matchedPts.push_back( kpKnown[matches[i].trainIdx].pt );
        }
    }

    /*vector<Point2f> kpCameraCorners(4); //TL, TR, BR, BL
    kpCameraCorners[0] = Point(0, 0);
    kpCameraCorners[1] = Point(rectifiedPresentationView.cols, 0);
    kpCameraCorners[2] = Point(rectifiedPresentationView.cols, rectifiedPresentationView.rows);
    kpCameraCorners[3] = Point(0, rectifiedPresentationView.rows);

    vector<Point2f> correlatedCorners(4);
    perspectiveTransform( kpCameraCorners, correlatedCorners, homography );
    */

    Mat rectifiedROI;
    cout << queryPts.size() << ", " << matchedPts.size() << endl;
    if (queryPts.empty() || queryPts.size() < 4) {
        return false;
    }
    Mat homography = findHomography( queryPts, matchedPts, RANSAC );
    warpPerspective( rectifiedPresentationView, rectifiedROI, homography, window_size);//rectifiedPresentationView.size() );

    imshow(rect_roi_image, rectifiedROI);

    /*
    averageXDist = averageXDistSum / numberOfGoodMatches;
    averageYDist = averageYDistSum / numberOfGoodMatches;

    cout << "Avg X dist: " << averageXDist << "\n";
    cout << "Avg Y dist: " << averageYDist << "\n";

    //-- Step 4: Adjust currently displayed mat by average distances
    // (assuming rectifiedCameraView has the same orientation as what's
    // currently displayed)

    double sizeDiffX =
            currentlyDisplayed.cols - rectifiedPresentationView.cols;
    if (sizeDiffX < 0) {
        sizeDiffX += averageXDist;
    } else {
        sizeDiffX = 0.0;
    }

    double sizeDiffY =
            currentlyDisplayed.rows - rectifiedPresentationView.rows;
    if (sizeDiffY < 0) {
        sizeDiffY += averageYDist;
    } else {
        sizeDiffY = 0.0;
    }

    Rect rectifiedROIRect(
            averageXDist, averageYDist,
            rectifiedPresentationView.cols - sizeDiffX,
            rectifiedPresentationView.rows - sizeDiffY
    );

    Mat rectifiedROI = rectifiedPresentationView.clone();
    rectifiedROI = rectifiedROI(rectifiedROIRect);*/

    //At this point, rectifiedROI should be the same size as currentlyDisplayed
    Mat diff;
    absdiff(rectifiedROI, currentlyDisplayed, diff);

    Mat hsvDiff;
    cvtColor(diff, hsvDiff, COLOR_BGR2HSV);

    Mat hopefullyLaser;
    inRange(hsvDiff, Scalar(0,180,180), Scalar(179, 255, 255), hopefullyLaser);

    imshow(window, hopefullyLaser);
}


void placeDotOnBoard(Point pointRelativeToBoard, Scalar bgrColor, bool continuing) {

    int lineWidth = 16;

    if (pointRelativeToBoard.x >= 0 && pointRelativeToBoard.y >= 0 &&
        pointRelativeToBoard.x < board.cols &&
        pointRelativeToBoard.y < board.rows) {

        if (continuing && lastPointOnBoard.x != -1 && lastPointOnBoard.y != -1) {
            line(board, lastPointOnBoard, pointRelativeToBoard, bgrColor, lineWidth);
        } else {
            circle(board, pointRelativeToBoard, lineWidth / 2, bgrColor, -1);
        }

        lastPointOnBoard = pointRelativeToBoard;
    }
}


/**
 * Pretty-prints the given int vector
 *
 * @param vec    the vector to print
 */
void printVector(vector<int> vec) {
    cout << "[ ";
    for (auto& item : vec) {
        cout << item << " ";
    }
    cout << "]" << endl;
}

/**
 * Gets average rgb value of the given mat (extracted blob) and returns its hsv value
 *
 * @param bgr    Image in BGR colorspace
 * @return       Returns HSV scalar that represents the average color of @param bgr
 */
Scalar BGRtoHSV(Mat bgr) {
    Scalar avg = mean(bgr);
    // create a 1x1 mat to convert to hsv using cvtColor()
    Mat bgr_one(1,1, CV_8UC3, avg);
    Mat hsv_one;
    cvtColor(bgr_one, hsv_one, CV_BGR2HSV);
    // extract the scalar from the 1x1 mat
    Scalar hsv = hsv_one.at<Vec3b>(Point(0, 0));
    cout << "HSV: " << hsv << endl;
    return hsv;
}

/**
 * Converts HSV scalar to BGR scalar
 *
 * @param hsv    HSV scalar to convert to BGR
 * @return       BGR scalar of @param hsv
 */
Scalar HSVtoBGR(Scalar hsv) {
    Mat hsv_one(1, 1, CV_8UC3, hsv);
    Mat bgr_one;
    cvtColor(hsv_one, bgr_one, CV_HSV2BGR);
    Scalar bgr = bgr_one.at<Vec3b>(Point(0, 0));
    return bgr;
}

/**
 * Helper function for sorting points in increasing y order.
 * Used as part of return points in TL, TR, BL, BR order.
 *
 * @param p1    Point 1
 * @param p2    Point 2
 * @return      True if p1.y < p2.y, else false.
 */
bool pointSorter(Point2f p1, Point2f p2) {
    return p1.y < p2.y;
}

/**
 * Draws red/blue circles at the specified points
 *
 * @param blue:    boolean that if true, the function draws blue circles. If false, draws red circles.
 */
void drawCircles(vector<Point2f> circles, bool blue, Mat image) {
    for (const auto& circ : circles) {
        blue ? circle(image, circ, calibration_circle_radius, Scalar(255, 0, 0), -1) :
        circle(image, circ, calibration_circle_radius, Scalar(0, 0, 255), -1);
    }
}

/**
 * Gets the four largest (in area) items from stats.
 * Only called if there are greater than 4 items in @param stats.
 *
 * @param nLabels    Number of items in @param stats
 * @param stats      The details of items from connectedComponentsWithStats
 *
 * @return           The centers of the four largest items in @param stats
 */
vector<Point2f> getFourLargest(int nLabels, Mat stats, int imgSize) {

    // precalculate upper blob size bound
    int blob_upper_bound = imgSize/8;

    // used for keeping track of which blobs have already been removed. Filled with ints: [0, nLabels)
    vector<int> idxes(nLabels);
    iota(begin(idxes), end(idxes), 0);

    // used for keeping track of the centers of the four largest blobs
    vector<Point2f> foundPoints;
    foundPoints.reserve(4);

    // find 4 blobs
    for (int numFound = 0; numFound < 4; numFound++) {
        int max_idx = -1;
        int max_area = -1;
        int pop_idx = -1;
        // find largest blob in indices specified in idxes
        for (int idx = 0; idx < idxes.size(); idx++) {
            int blob_area = stats.at<int>(idxes[idx], CC_STAT_AREA);
            if (blob_area > max_area && blob_area < blob_upper_bound) {
                max_idx = idxes[idx];
                max_area = blob_area;
                pop_idx = idx;
            }
        }
        // create a point at center of this round's biggest blob
        Point2f foundPoint = Point2f(
                stats.at<int>(max_idx, CC_STAT_LEFT) + stats.at<int>(max_idx, CC_STAT_WIDTH)/2,
                stats.at<int>(max_idx, CC_STAT_TOP) + stats.at<int>(max_idx, CC_STAT_HEIGHT)/2
        );
        // store the new point
        foundPoints.push_back(foundPoint);
        cout << "Found idx " << max_idx << ": " << foundPoint << endl;
        //if (numFound != 3) {
        idxes.erase(idxes.begin() + pop_idx);
        //}
        // printVector(idxes);
    }
    return foundPoints;
}


/**
 * Given a vector of input frames, finds the areas where there is consistent change in the blue/red channels.
 *
 * @param frames     Vector of frames to process
 * @return           Returns a vector of points in the order TL, TR, BR, BL
 */
vector<Point2f> getDifferences(vector<Mat> frames) {
    // used to keep track of areas of interest
    Mat storedCircles(frames[0].rows, frames[0].cols, CV_8UC1, 255);
    // iterate over given frames
    for (size_t idx = 0; idx < frames.size() - 1; idx++) {
        Mat diff;
        absdiff(frames[idx], frames[idx+1], diff);
        vector<Mat> channels;
        // extract channels from diff image
        split(diff, channels);

        // combine blue and red channels
        Mat comp = channels[0] + channels[2];
        // keep pixels above a certain diff threshold
        comp = comp > pic_diff;
        //imshow(difference_window, comp);
        // keep track of only the areas that consistently show red/blue change
        bitwise_and(storedCircles, comp, storedCircles);
    }
    imshow(difference_window, storedCircles);

    // Process the image to highlight the blobs we want.
    // some noise removal?
    medianBlur(storedCircles, storedCircles, 5);
    // blobify!
    for(int num_dilations = 0; num_dilations < curr_num_dilations; num_dilations++) {
        dilate(storedCircles, storedCircles, getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point2f(-1, -1)));
    }

    // extract blobs
    Mat stats, centroids, labelImage;
    int nLabels = connectedComponentsWithStats(storedCircles, labelImage, stats, centroids, 8);

    // if we didn't find at least 4 blobs, increase num dilations and try again with 6 new frames
    if (nLabels < 4) {
        curr_num_dilations++;
        return vector<Point2f>{};
    }
    else if (nLabels > 15) {
        return vector<Point2f>{};
    }
    else {
        cout << "Number of blobs: " << nLabels << endl;
        vector<Point2f> points = getFourLargest(nLabels, stats, storedCircles.rows * storedCircles.cols);
        return points;
    }

    /*
    while (nLabels < 4) {
        cout << "Dilating and rerunning connectedComponents" << endl;
        dilate(storedCircles, storedCircles, getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point2f(-1, -1)));
        nLabels = connectedComponentsWithStats(storedCircles, labelImage, stats, centroids, 8);
        waitKey(500);
    }
    */
}

// TODO: Return corners in order: TL, TR, BR, BL
/**
 * The main calibration loop. The first cycle allows the user to position the laptop at the projected screen.
 * The second cycle is the actual calibration cycle. It places a circle in each corner of the image, and flashes them
 * alternatingly red and blue. It then extracts the locations of these circles from the webcam image, allowing the
 * skew/warp to be calculated.
 *
 * @return
 */
Mat getTransformationMatrix() {
    // create the calibration image and display it
    Mat testImage(window_size, CV_8UC3, Scalar(255, 255, 255));

    int edgeMargin = 25;

    //Create corners (centers of circles)
    Point2f topLeft(edgeMargin, edgeMargin);
    Point2f topRight(testImage.cols - edgeMargin, edgeMargin);
    Point2f bottomRight(testImage.cols - edgeMargin, testImage.rows - edgeMargin);
    Point2f bottomLeft(edgeMargin, testImage.rows - edgeMargin);

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

    // this window needs to be put on the projection screen
    namedWindow(calibration_window, CV_WINDOW_NORMAL);
    imshow(calibration_window, testImage);

    // create the webcam capture and prep for displaying it
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening webcam" << endl;
        return Mat(0,0,CV_8UC1);
    }
    namedWindow(webcam_window, CV_WINDOW_NORMAL);
    //setWindowProperty(webcam_window, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

    // show calibration window and webcam image until user says everything is positioned
    cout << "Press q to start calibration..." << endl;
    bool blue = false;
    Mat currFrame;
    while ((char)waitKey(40) != 'q') {
        blue = !blue;
        drawCircles(points, blue, testImage);
        cap >> currFrame;
        imshow(calibration_window, testImage);
        imshow(webcam_window, currFrame);
    }

    cout << "Starting calibration..." << endl;

    // setup display for difference display
    namedWindow(difference_window, CV_WINDOW_NORMAL);

    // setup window for displaying found points
    namedWindow(found_points_window, CV_WINDOW_NORMAL);

    // find the points
    vector<Point2f> found_points;
    vector<Mat> frames;
    // grab num_cap_frames at a time and send to getDifferences for processing, alternating between red and blue
    while (found_points.empty() || found_points.size() != 4) {
        for(int i = 0; i < num_cap_frames; i++) {
            blue = !blue;
            drawCircles(points, blue, testImage);
            cap >> currFrame;
            // frames.push_back(testImage.clone());
            frames.push_back(currFrame.clone());
            imshow(calibration_window, testImage);
            imshow(webcam_window, currFrame);
            waitKey(100);
        }
        found_points = getDifferences(frames); // displays overall binary difference on differences_window
        // display green rectangles around the found points
        Mat found_points_mat = currFrame.clone();
        for (Point2f point : found_points) {
            rectangle(found_points_mat, Point2f(point.x - 15, point.y - 15), Point2f(point.x + 15, point.y + 15), Scalar(0,255,0), 2);
        }
        imshow(found_points_window, found_points_mat);
    }
    cout << "Calibration complete." << endl;
    cap.release();

    // sort in increasing y order
    sort(found_points.begin(), found_points.end(), pointSorter);
    // use x-values to check TL, TR, BR, BL order
    if (found_points[0].x > found_points[1].x) {
        swap(found_points[0], found_points[1]);
    }
    if (found_points[3].x > found_points[2].x) {
        swap(found_points[2], found_points[3]);
    }

    // print the points
    for (auto& point : found_points) {
        cout << point << endl;
    }

    //Mat transformationMatrix = findHomography(found_points, points);
    Mat transformationMatrix = getPerspectiveTransform(found_points, points);
    return transformationMatrix;
}

int transformWebcamImage(const Mat transformationMatrix) {
    // temporary display image
    //Mat tempDisplay(window_size, CV_8UC3, Scalar(255,255,255));
    Mat dewarpedWebcam;


    // showing points for testing purposes
    int edgeMargin = 25;
    Point2f topLeft(edgeMargin, edgeMargin);

    Point2f topRight(board.cols - edgeMargin, edgeMargin);
    Point2f bottomRight(board.cols - edgeMargin, board.rows - edgeMargin);
    Point2f bottomLeft(edgeMargin, board.rows - edgeMargin);

    vector<Point2f> points; //Clockwise: TL, TR, BR, BL
    points.push_back(topLeft);
    points.push_back(topRight);
    points.push_back(bottomRight);
    points.push_back(bottomLeft);
    bool blue = true;
    drawCircles(points, blue, board);

    // start the webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening webcam" << endl;
        return -1;
    }

    Mat currFrame;
    while ((char)waitKey(40) != 'q') {
        cap >> currFrame;
        warpPerspective(currFrame, dewarpedWebcam, transformationMatrix, window_size);
        flip(dewarpedWebcam, dewarpedWebcam, 1);
        imshow(webcam_window, dewarpedWebcam);
        imshow(calibration_window, board);
        Point2f laserLoc;
        getLocationOfLaserPoint(dewarpedWebcam, board, laserLoc);
        waitKey(40);
    }
    return 0;
}

void initBoard(Size pixels) {
    board = Mat(pixels, CV_8UC3, Scalar(255,255,255));
}

int main(int argc, char** argv) {
    Mat transformationMatrix = getTransformationMatrix();
    cout << transformationMatrix << endl;

    // destroy unnecessary windows
    destroyWindow(found_points_window);
    destroyWindow(difference_window);
    //destroyWindow(webcam_window);

    // start processing for laser pointer
    initBoard(window_size);
    placeDotOnBoard(Point(300, 300), Scalar(0,0,255), false);
    placeDotOnBoard(Point(400, 300), Scalar(0,0,255), false);
    placeDotOnBoard(Point(200, 400), Scalar(0,0,255), false);
    placeDotOnBoard(Point(350, 450), Scalar(0,0,255), true);
    placeDotOnBoard(Point(500, 400), Scalar(0,0,255), true);
    namedWindow(window, CV_WINDOW_NORMAL);
    namedWindow(rect_roi_image, CV_WINDOW_NORMAL);

    int retval = transformWebcamImage(transformationMatrix);
    if (retval) {
        return EXIT_FAILURE;
    }
    else {
        return EXIT_SUCCESS;
    }
}

/*
//get fps (https://www.learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/)
int main(int argc, char** argv)
{

    // Start default camera
    VideoCapture video(0);

    // With webcam get(CV_CAP_PROP_FPS) does not work.
    // Let's see for ourselves.

    double fps = video.get(CV_CAP_PROP_FPS);
    // If you do not care about backward compatibility
    // You can use the following instead for OpenCV 3
    // double fps = video.get(CAP_PROP_FPS);
    cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;


    // Number of frames to capture
    int num_frames = 120;

    // Start and end times
    time_t start, end;

    // Variable for storing video frames
    Mat frame;

    cout << "Capturing " << num_frames << " frames" << endl ;

    // Start time
    time(&start);

    // Grab a few frames
    for(int i = 0; i < num_frames; i++)
    {
        video >> frame;
    }

    // End Time
    time(&end);

    // Time elapsed
    double seconds = difftime (end, start);
    cout << "Time taken : " << seconds << " seconds" << endl;

    // Calculate frames per second
    fps  = num_frames / seconds;
    cout << "Estimated frames per second : " << fps << endl;

    // Release video
    video.release();
    return 0;
}*/