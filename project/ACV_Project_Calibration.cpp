/**
 * Ben Brown : beb5277
 * Mark Philip : msp3430
 * Advanced Computer Vision Project
 *
 * TODO: add accurate description
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>

#include <iostream>

// #include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

// drawing constants
const int calibration_circle_radius = 20;

// filter params
int pic_diff = 2;
int num_cap_frames = 6;

// window names
const char *calibration_window = "Calibration Window";
const char *webcam_window = "Webcam Image";
const char *difference_window = "Calibration";

/**
 * Gets average rgb value of the given mat (extracted blob) and returns its hsv value
 *
 * @param bgr    Image in BGR colorspace
 * @return       Returns HSV scalar that represents the average color of @param bgr
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
 * Given a vector of input frames, finds the areas where there is consistent change in the blue/red channels.
 *
 * @param frames     Vector of frames to process
 * @return           Returns a vector of keypoints in the order TL, TR, BR, BL
 */
vector<KeyPoint> getDifferences(vector<Mat> frames) {
    // used to keep track of areas of interest
    Mat storedCircles(frames[0].rows, frames[0].cols, CV_8UC1, 255);
    Mat diffMat(frames[0].rows, frames[0].cols, DataType<int>::type);
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

    /*
    // Used originally for finding a good pic_diff value
    if (countNonZero(diff) > (diff.rows * diff.cols)/8) {
        pic_diff++;
        cout << "Difference threshold: " << pic_diff << endl;
    }
    else if (pic_diff > 1){
        pic_diff--;
    }
     */

    /* FLOW
     * ----
     * Dilate (x2?)
     * Find blobs (connected compononets w/ stats?) filter by a max size
     *    - should be 4 of them, otherwise get 4 largest
     *    - ensure that there is 1 per quadrant
     *       - if not, rerun with a new frame cap count?
     */
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

    // this window needs to be put on the projection screen
    namedWindow(calibration_window, CV_WINDOW_NORMAL);
    imshow(calibration_window, testImage);

    // create the webcam capture and prep for displaying it
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening webcam" << endl;
        return vector<KeyPoint>{};
    }
    namedWindow(webcam_window, CV_WINDOW_NORMAL);
    setWindowProperty(webcam_window, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

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
    vector<KeyPoint> num_points;
    vector<Mat> frames;
    // grab num_cap_frames at a time and send to getDifferences for processing, alternating between red and blue
    while (num_points.empty() || num_points.size() != 4) {
        for(int i = 0; i < num_cap_frames; i++) {
            blue = !blue;
            drawCircles(points, blue, testImage);
            cap >> currFrame;
            // frames.push_back(testImage.clone());
            frames.push_back(currFrame.clone());
            imshow(calibration_window, testImage);
            imshow(webcam_window, currFrame);
            waitKey(40);
        }
        num_points = getDifferences(frames); // displays overall binary difference on differences_window
    }
    cout << "Calibration complete." << endl;
    cap.release();
    return num_points;
}


int main(int argc, char** argv) {
    getCorners();
    return EXIT_SUCCESS;
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