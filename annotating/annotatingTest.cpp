/*
 * Just an attempt at an mspaint clone using OpenCV
 */

#include <opencv/cv.hpp>
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
int circleLine = -1;
// line width (2x circle radius)
int lineWidth = 10;

// mouse callback
void onMouse(int event, int x, int y, int flags, void* data) {
    switch (event) {
        case EVENT_MOUSEMOVE:
            if (mouseDown) {
                // cout << "mouse down: (" << x << ", " << y << ")" << endl;
                circle(*((Mat *)data), Point(x, y), int(ceil(lineWidth/2.0)), Scalar(0, 0, 255), circleLine);
            }
            else {
                cout << "mouse up: (" << x << ", " << y << ")" << endl;
            }
            break;
        case EVENT_LBUTTONDOWN:
            mouseDown = true;
            break;
        case EVENT_LBUTTONUP:
            mouseDown = false;
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
        setMouseCallback(windowName, onMouse, &img);
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