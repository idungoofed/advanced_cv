/*
 * Just an attempt at an mspaint clone using OpenCV
 */

#include <opencv/cv.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>


using namespace cv;
using namespace std;

/* TODO: Drawing ideas:
 * - Trackbar for line size
 * - undo queue
 */

// flag for if the mouse is pressed
bool mouseDown = false;

// circle line thickness (-1 = filled circle)
int circleLineThickness = -1;

// max line width
const int lineWidthMax = 100;
// line width (2x circle radius)
int lineWidth = 10;

// line color
Scalar lineColor = Scalar(0, 0, 255);

// keeps track of previous point for line drawing
Point prevPoint = Point(-1, -1);

// trackbar callback
/*void onTrackbar(int trackbarPos, void* data) {
    cout << "\"" << trackbarPos << "\"" << endl;
}*/


// mouse callback
void onMouse(int event, int x, int y, int flags, void* data) {
    switch (event) {
        case EVENT_MOUSEMOVE:
            if (mouseDown) {
                // cout << "mouse down: (" << x << ", " << y << ")" << endl;
                Point curPoint = Point(x, y);
                if (prevPoint.x != -1) {
                    line(*((Mat *)data), prevPoint, curPoint, lineColor, lineWidth);
                }
                else {
                    circle(*((Mat *)data), Point(x, y), int(ceil(lineWidth/2.0)), lineColor, circleLineThickness);
                }
                prevPoint = curPoint;
            }
            break;
        case EVENT_LBUTTONDOWN:
            mouseDown = true;
            break;
        case EVENT_LBUTTONUP:
            mouseDown = false;
            prevPoint = Point(-1, -1);
            break;
        default:
            cout << "Other event: " << event << endl;
            break;
    }
}


int main(int argc, char* argv[]) {
    string usageMessage = "usage: AnnotatingTest image_file_name";

    // ensure correct number of args
    if (argc != 2) {
        cout << "Error: Incorrect number of arguments" << endl << usageMessage << endl;
        return EXIT_FAILURE;
    }

    // try to load the image, ensure validity
    Mat img = imread(argv[1], CV_LOAD_IMAGE_ANYCOLOR);
    if (img.data) {
        // start annotating
        String windowName = "Testing";
        namedWindow(windowName, WINDOW_AUTOSIZE);
        resizeWindow(windowName, 800, 800);
        moveWindow(windowName, 10, 10);

        // trackbar setup
        createTrackbar("Line Size", windowName, &lineWidth, lineWidthMax);//, onTrackbar);

        // mouse callback
        setMouseCallback(windowName, onMouse, &img);

        // main loop
        while ((char)waitKey(1) != 'q') {
            imshow(windowName, img);
        }
        return EXIT_SUCCESS;
    }
    else {
        cout << "Error: Invalid image file" << endl << usageMessage << endl;
        return EXIT_FAILURE;
    }
}