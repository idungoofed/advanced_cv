/**
 * Ben Brown : beb5277
 * Mark Philip : msp3430
 * Advanced Computer Vision Project
 *
 * Maps a webcam image of a projection back onto the original image that is being projected, allowing for a user with
 * a laser pointer to interact with the laptop just by pointing the laser at the projector screen.
 *
 * How to use:
 * First the computer must be in "extended display" mode, connected to a projector.
 * After starting the program, move the "Calibration Window" over to the extended projector display and fullscreen it.
 * Position the laptop so that the four dots in the corners of the projected screen show up in the webcam view.
 * Once the laptop is positioned properly (i.e. all four dots are visible), press 'q' on the keyboard to start the
 * calibration, which should only take a few seconds. Once the calibration is complete, a (poorly-drawn) smiley face
 * will appear on the screen. You can then start drawing on the screen using a green laser pointer pointed at the
 * projected image.
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv/cv.hpp>

#include <numeric>
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// drawing constants
const int calibration_circle_radius = 20;
const Size window_size = Size(1024,768);
const int line_size = 10;
const int edgeMargin = 25;
const int LINE_DRAW_DISTANCE_CUTOFF = 20; //px

// filter params
const int pic_diff = 3;
const int num_cap_frames = 18;
int curr_num_dilations = 3;
const int lower_laser_bound = 200;
const int min_blob_area = 500;
const int LASER_DILATE_KERNEL_WIDTH = 6; //px

// window names
const char *calibration_window = "Calibration Window";
const char *webcam_window = "Webcam Image";
const char *difference_window = "Processing Frames for Corners";
const char *found_points_window = "Found Corners";
const char *window = "Processing for Laser";

// Mat used for drawing on with laser
Mat board;

// Used for drawing on board
Point lastPointOnBoard(-1, -1);
bool firstPoint = false;

// Used for processing to find laser in image
Mat kernel = getStructuringElement(
        MORPH_ELLIPSE, Size( LASER_DILATE_KERNEL_WIDTH, LASER_DILATE_KERNEL_WIDTH ),
        Point( -1,-1 )
);
Mat kernel2 = getStructuringElement(
        MORPH_ELLIPSE, Size( LASER_DILATE_KERNEL_WIDTH*2, LASER_DILATE_KERNEL_WIDTH*2),
        Point( -1, -1)
);

/**
 * Given the dewared webcam view, finds the location of a laser pointer in the image and stores it in @param
 * laserPointOut and returns true. If no laser is present, returns false.
 *
 * @param rectifiedPresentationView This is the dewarped image coming from the webcam of the projection screen
 * @param laserPointOut If a laser point is found, this is provided the X and Y location of the laser point
 *
 * @return True if a laser is present (with the location stored in @param laserPointOut).
 *         Else false.
 */
bool getLocationOfLaserPoint(Mat rectifiedPresentationView, Point2i &laserPointOut) {
    Mat hopefullyLaser;
    // Use thresholding to find the (super-bright) laser.
    inRange(rectifiedPresentationView,
            Scalar(lower_laser_bound, lower_laser_bound, lower_laser_bound),
            Scalar(255, 255, 255), hopefullyLaser
    );

    // Get rid of most noise.
    dilate(hopefullyLaser, hopefullyLaser, kernel);
    morphologyEx(hopefullyLaser, hopefullyLaser, MORPH_OPEN, kernel2);
    imshow(window, hopefullyLaser);

    // extract all blobs
    Mat ccLabels, ccStats, ccCentroids;
    int numCCs = connectedComponentsWithStats(
            hopefullyLaser, ccLabels, ccStats, ccCentroids, 8
    );
    /*
    if (numCCs > 20) {
        return false;
    }
     */

    // find largest blob
    int max_idx = -1;
    int max_area = -1;
    for (int idx = 0; idx < numCCs; idx++) {
        int blob_area = ccStats.at<int>(idx, CC_STAT_AREA);
        if (blob_area > min_blob_area && blob_area > max_area && blob_area < (window_size.height * window_size.width) / 10) {
            max_idx = idx;
            max_area = blob_area;
        }
    }

    if (max_idx == -1) {
        // No suitable blob was found
        return false;
    } else {
        // Suitable blob was found, so store its location in laserPointOut and return true.
        int ccHeight = ccStats.at<int>(max_idx, CC_STAT_HEIGHT);
        int ccWidth = ccStats.at<int>(max_idx, CC_STAT_WIDTH);
        int y = ccStats.at<int>(max_idx, CC_STAT_TOP);
        int x = ccStats.at<int>(max_idx, CC_STAT_LEFT);
        laserPointOut.x = x + ccWidth / 2;
        laserPointOut.y = y + ccHeight / 2;
        return true;
    }
}

/**
 * Places a point on the board at the specified point (@param pointRelativeToBoard). Determines whether to draw a line
 * or just a point.
 *
 * @param pointRelativeToBoard The point on the board to draw at
 * @param bgrColor The color to draw with (BGR)
 * @param continuing Draw a line from the last point to this one? Otherwise, draw a single dot.
 */
void placeDotOnBoard(Point pointRelativeToBoard, Scalar bgrColor, bool continuing) {

    int lineWidth = line_size;

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
 * Helper function for sorting points in increasing y order.
 * Used as part of putting points in TL, TR, BL, BR order.
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
 * @param blue    boolean that if true, the function draws blue circles. If false, draws red circles.
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

    // find 4 largest blobs
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
        // create a point at center of this loop's biggest blob
        Point2f foundPoint = Point2f(
                stats.at<int>(max_idx, CC_STAT_LEFT) + stats.at<int>(max_idx, CC_STAT_WIDTH)/2,
                stats.at<int>(max_idx, CC_STAT_TOP) + stats.at<int>(max_idx, CC_STAT_HEIGHT)/2
        );
        // store the new point
        foundPoints.push_back(foundPoint);
        //cout << "Found idx " << max_idx << ": " << foundPoint << endl;
        // remove this blob from further rounds of comparison
        idxes.erase(idxes.begin() + pop_idx);
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
        imshow(difference_window, comp);
        // keep track of only the areas that consistently show red/blue change
        bitwise_and(storedCircles, comp, storedCircles);
    }
    imshow(difference_window, storedCircles);

    // Process the image to highlight the blobs we want.
    // Remove some single-pixel noise
    medianBlur(storedCircles, storedCircles, 5);
    // blobify!
    for(int num_dilations = 0; num_dilations < curr_num_dilations; num_dilations++) {
        dilate(storedCircles, storedCircles, getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point2f(-1, -1)));
    }

    // extract blobs
    Mat stats, centroids, labelImage;
    int nLabels = connectedComponentsWithStats(storedCircles, labelImage, stats, centroids, 8);

    // if we didn't find at least 4 suitable blobs, increase num_dilations and try again with a set of new frames
    if (nLabels < 4) {
        curr_num_dilations++;
        // cout << "Increasing curr_num_dilations to " << curr_num_dilations << endl;
        return vector<Point2f>{};
    }
    else if (nLabels > 15) {
        // Too many blobs, most likely because something moved. Try again with a new set of frames.
        return vector<Point2f>{};
    }
    else {
        // cout << "Number of blobs: " << nLabels << endl;
        // get the four largest blobs, which should be the corner blobs (i.e., the only blobs we want)
        vector<Point2f> points = getFourLargest(nLabels, stats, storedCircles.rows * storedCircles.cols);
        return points;
    }
}

/**
 * The main calibration loop. The first cycle allows the user to position the laptop at the projected screen.
 * The second cycle is the actual calibration cycle. It places a circle in each corner of the image, and flashes them
 * alternatingly red and blue. It then extracts the locations of these circles from the webcam image, allowing the
 * skew/warp to be calculated.
 *
 * @return    The transformation matrix for removing the detected warp.
 */
Mat getTransformationMatrix() {
    // create the calibration image and display it
    Mat testImage(window_size, CV_8UC3, Scalar(255, 255, 255));

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
    /*
    cout << "Original point locations:" << endl;
    cout << "\tTL: " << topLeft << ", " << endl;
    cout << "\tTR: " << topRight << ", " << endl;
    cout << "\tBL: " << bottomLeft << ", " << endl;
    cout << "\tBR: " << bottomRight << ", " << endl;
    */
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

    // show calibration window and webcam image until user indicates everything is positioned properly
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
            frames.push_back(currFrame.clone());
            imshow(calibration_window, testImage);
            imshow(webcam_window, currFrame);
            waitKey(100);
        }
        found_points = getDifferences(frames);
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

    // use x-values to check for TL, TR, BR, BL order
    if (found_points[0].x > found_points[1].x) {
        swap(found_points[0], found_points[1]);
    }
    if (found_points[3].x > found_points[2].x) {
        swap(found_points[2], found_points[3]);
    }

    // print the points
    cout << "Found points:" << endl;
    for (auto& point : found_points) {
        cout << point << endl;
    }

    // Create the transformation matrix by mapping our extracted points back to their original counterparts.
    Mat transformationMatrix = getPerspectiveTransform(found_points, points);
    return transformationMatrix;
}

/**
 * Main loop for laser detection and drawing.
 * Captures frames from the computer's webcam, dewarps them using the precalculated transformation matrix, and then
 * detects if a laser is present in the image. If it is present, it draws the corresponding point on the projected image.
 *
 * @param transformationMatrixX
 * @param transformationMatrixY
 *
 * @return                         Non-zero integer if there is an error, otherwise 0 to indicate successful run.
 */
int transformWebcamImage(Mat transformationMatrixX, Mat transformationMatrixY) {
    Mat dewarpedWebcam;

    // showing points for testing purposes
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

    // Check each frame for a laser
    cout << "Starting whiteboard mode..." << endl;
    cout << "Press 'q' to quit." << endl;
    Mat currFrame;
    while ((char)waitKey(1) != 'q') {
        cap >> currFrame;

        remap(currFrame, dewarpedWebcam, transformationMatrixX,
              transformationMatrixY, INTER_LINEAR);
        //imshow(webcam_window, dewarpedWebcam);
        Point2i laserLoc;
        if (getLocationOfLaserPoint(dewarpedWebcam, laserLoc)) {
            // Found single point, so draw it to the image

            bool drawLine = false;
            if (firstPoint && norm(lastPointOnBoard-laserLoc) <= LINE_DRAW_DISTANCE_CUTOFF) {
                drawLine = true;
            }
            placeDotOnBoard(laserLoc, Scalar(0, 0, 255), drawLine);
            if (!firstPoint) {
                firstPoint = true;
            }
        }
        // display the drawn-on image
        imshow(calibration_window, board);
    }
    return 0;
}

/**
 * Generates premapped matrices for performance inspired by 
 * http://romsteady.blogspot.in/2015/07/calculate-opencv-warpperspective-map.html
 *
 * @param transformationMatrix The original transformation matrix, provided by getPerspectiveTransform
 * @param frameSize The size to make the final matrix
 * @param outTransformationX Transformation matrix which will be filled with the premapped points, 
                             pertaining to the X transformations of the transformationMatrix
 * @param outTransformationY Transformation matrix which will be filled with the premapped points, 
                             pertaining to the Y transformations of the transformationMatrix
 */
void createPremappedWarp(Mat transformationMatrix, Size frameSize,
                         Mat &outTransformationX, Mat &outTransformationY) {

    // Since the camera won't be moving, let's pregenerate the remap LUT
    cv::Mat inverseTransMatrix;
    cv::invert(transformationMatrix, inverseTransMatrix);

    // Generate the warp matrix
    cv::Mat map_x, map_y, srcTM;
    srcTM = inverseTransMatrix.clone(); // If WARP_INVERSE, set srcTM to transformationMatrix

    map_x.create(frameSize, CV_32FC1);
    map_y.create(frameSize, CV_32FC1);

    double M11, M12, M13, M21, M22, M23, M31, M32, M33;
    M11 = srcTM.at<double>(0,0);
    M12 = srcTM.at<double>(0,1);
    M13 = srcTM.at<double>(0,2);
    M21 = srcTM.at<double>(1,0);
    M22 = srcTM.at<double>(1,1);
    M23 = srcTM.at<double>(1,2);
    M31 = srcTM.at<double>(2,0);
    M32 = srcTM.at<double>(2,1);
    M33 = srcTM.at<double>(2,2);

    for (int y = 0; y < frameSize.height; y++) {
        double fy = (double)y;
        for (int x = 0; x < frameSize.width; x++) {
            double fx = (double)x;
            double w = ((M31 * fx) + (M32 * fy) + M33);
            w = w != 0.0f ? 1.f / w : 0.0f;
            float new_x = (float)((M11 * fx) + (M12 * fy) + M13) * w;
            float new_y = (float)((M21 * fx) + (M22 * fy) + M23) * w;
            map_x.at<float>(y,x) = new_x;
            map_y.at<float>(y,x) = new_y;
        }
    }

    // This creates a fixed-point representation of the mapping resulting in ~4% CPU savings
    outTransformationX.create(frameSize, CV_16SC2);
    outTransformationY.create(frameSize, CV_16UC1);
    cv::convertMaps(map_x, map_y, outTransformationX, outTransformationY, false);
}

/**
 * Initializes board to be a 3-channel matrix of size @param pixels, with each pixel having an
 * initial value of (255,255,255).
 *
 * @param pixels    The size to make board
 */
void initBoard(Size pixels) {
    board = Mat(pixels, CV_8UC3, Scalar(255,255,255));
}

/**
 * First starts the auto-calibration to get the transformation matrix for dewarping, then passes that matrix to the
 * function that detects a laser's presence in the webcam image for drawing purposes.
 *
 * @param argc    unused
 * @param argv    unused
 * @return        EXIT_FAILURE if error, otherwise EXIT_SUCCESS
 */
int main(int argc, char** argv) {
    // get the transformation matrix
    Mat transformationMatrix = getTransformationMatrix();
    // cout << transformationMatrix << endl;

    // destroy unnecessary windows
    destroyWindow(found_points_window);
    destroyWindow(difference_window);
    destroyWindow(webcam_window);

    // Start processing for laser pointer. Pre-draw a smiley-face.
    initBoard(window_size);
    //placeDotOnBoard(Point(300, 300), Scalar(0,0,255), false);
    //placeDotOnBoard(Point(400, 300), Scalar(0,0,255), false);
    //placeDotOnBoard(Point(200, 400), Scalar(0,0,255), false);
    //placeDotOnBoard(Point(350, 450), Scalar(0,0,255), true);
    //placeDotOnBoard(Point(500, 400), Scalar(0,0,255), true);
    namedWindow(window, CV_WINDOW_NORMAL);

    Mat mappedTransformX, mappedTransformY;

    createPremappedWarp(transformationMatrix, window_size,
                        mappedTransformX, mappedTransformY);
    int retval = transformWebcamImage(mappedTransformX, mappedTransformY);
    cout << "DONE" << endl;
    if (retval) {
        return EXIT_FAILURE;
    }
    else {
        return EXIT_SUCCESS;
    }
}