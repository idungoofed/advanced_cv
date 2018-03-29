// Created by mark (msp3430) on 3/8/18.

//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

//C/C++
#include <iostream>
#include <dirent.h>

using namespace cv;
using namespace std;


int processVideo(VideoCapture vid);

int main(int argc, char *argv[]) {

    // get list of files in ./videos/
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir("./videos/"))) {
        while (ent = readdir(dir)) {
            // exclude . and ..
            if (strcmp(ent->d_name, ".\0") && strcmp(ent->d_name, "..\0")) {

                // create the filename string (./videos/<filename>)
                char *filename = (char *) malloc(sizeof(char) * (10 + strlen(ent->d_name)));
                filename = strncpy(filename, "./videos/\0", 10 + strlen(ent->d_name));
                filename = strcat(filename, ent->d_name);

                // open the video file
                printf("Processing \"%s\"\n", filename);
                VideoCapture vid(filename);
                // Check if camera opened successfully
                if (!vid.isOpened()) {
                    std::cout << "Error! Can't open video file: " << filename << std::endl;
                } else {
                    processVideo(vid);
                    // cleanup
                    vid.release();
                }
                // free the earlier-malloced filename
                free(filename);
            }
        }
        closedir(dir);
    } else {
        // could not open directory
        std::cout << "Error! ./videos/ doesn't exist." << std::endl;
        return EXIT_FAILURE;
    }
    destroyAllWindows();
    return EXIT_SUCCESS;
}


int processVideo(VideoCapture vid) {

    // create the mog2 object: history=100 frames, shadow_detection=false
    Ptr<BackgroundSubtractor> mog2;
    mog2 = createBackgroundSubtractorMOG2(100, 16, false);

    // current frame
    Mat frame;
    //fg mask generated by MOG2
    Mat fgMaskMOG2;



    // set up the display windows
    string original_vid_window = "Input Video";
    string fg_mask_window = "Foreground Mask (MOG2)";
    // commented out displaying training

    namedWindow(original_vid_window, WINDOW_FREERATIO);
    namedWindow(fg_mask_window, WINDOW_FREERATIO);
    // resize to fit nicely on my screen, keeping aspect ratio the same
    resizeWindow(original_vid_window, 360, 640);
    resizeWindow(fg_mask_window, 360, 640);
    // move to a nice spot on my screen
    //moveWindow(original_vid_window, 1700, 0);
    //moveWindow(fg_mask_window, 2061, 10);
    moveWindow(original_vid_window, 0, 0);
    moveWindow(fg_mask_window, 361, 10);


    // Read input video.
    // Train MOG2 on the first 500 frames, skipping every other frame (so actually 250 frames)
    // MOG2 initialized with memory of 100 frames, so really only the last 200 frames matter.
    int skip = 1;
    for (int frameCount = 0; frameCount < 500; frameCount++) {
        //skip every other frame
        skip = !skip;
        if (skip) {
            continue;
        }

        //read the current frame, break on error (or end of file)
        if (!vid.read(frame)) {
            cerr << "End of video reached." << endl;
            break;
        }

        // gaussian blur the frame before feeding into MOG2 to get rid of some noise
        //Mat blurred_frame;
        //GaussianBlur(frame, blurred_frame, Size(9, 9), 0, 0);
        //update the background model
        mog2->apply(frame, fgMaskMOG2);

        // commented out displaying video during training

        //show the current frame and the fg mask
        imshow(original_vid_window, frame);
        imshow(fg_mask_window, fgMaskMOG2);
        waitKey(1);

    }

    // setup to display the background model
    string mog2_background = "Background Model (press q to continue)";
    namedWindow(mog2_background, WINDOW_FREERATIO);
    // resize to fit nicely on my screen, keeping aspect ratio the same
    resizeWindow(mog2_background, 360, 640);
    //moveWindow(mog2_background, 2420, 0);
    moveWindow(mog2_background, 722, 0);

    // get and display the background model
    Mat background;
    mog2->getBackgroundImage(background);
    cout << "Displaying background model (press q to continue)" << endl;
    imshow(mog2_background, background);
    // wait until user presses 'q' to move on
    while ((char)waitKey(0) != 'q') {}
    destroyAllWindows();

    cout << "Drawing bounding boxes..." << endl;

    namedWindow(original_vid_window, WINDOW_FREERATIO);
    namedWindow(fg_mask_window, WINDOW_FREERATIO);
    // resize to fit nicely on my screen, keeping aspect ratio the same
    resizeWindow(original_vid_window, 360, 640);
    resizeWindow(fg_mask_window, 360, 640);
    // move to a nice spot on my screen
    //moveWindow(original_vid_window, 1700, 0);
    //moveWindow(fg_mask_window, 2061, 10);
    moveWindow(original_vid_window, 0, 0);
    moveWindow(fg_mask_window, 361, 10);


    // reset to beginning of video
    vid.set(CV_CAP_PROP_POS_FRAMES, 0);

    // used for keeping track of previously drawn rectangles, initialized to zeros
    Mat previous_rects = Mat::zeros(background.size(), CV_8U);

    // used for keeping track of the last-drawn rectangle
    Rect prev_rect;
    // set so that we can check if it was initialized
    prev_rect.width = -1;
    Rect cur_rect;

    skip = 1;
    while(1) {
        //skip every other frame
        skip = !skip;
        if (skip) {
            continue;
        }

        //read the current frame, break on error (or more likely end of file)
        if (!vid.read(frame)) {
            cerr << "End of video reached." << endl;
            break;
        }

        //update the background model
        mog2->apply(frame, fgMaskMOG2);

        // gaussian blur to get rid of noise
        //GaussianBlur(fgMaskMOG2, fgMaskMOG2, Size(9, 9), 0, 0);

        // dilate first to make the diver a more-cohesive blob
        dilate(fgMaskMOG2, fgMaskMOG2, getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1)));

        // erode to get rid of some noise
        erode(fgMaskMOG2, fgMaskMOG2, getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1)));
        erode(fgMaskMOG2, fgMaskMOG2, getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1)));

        // dilate to blobify the diver
        dilate(fgMaskMOG2, fgMaskMOG2, getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1)));
        dilate(fgMaskMOG2, fgMaskMOG2, getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1)));
        dilate(fgMaskMOG2, fgMaskMOG2, getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1)));

        // get blobs
        Mat stats, centroids, labelImage;
        int nLabels = connectedComponentsWithStats(fgMaskMOG2, labelImage, stats, centroids, 8);
        int maxArea = 0;
        int maxIndex = -1;
        for (int idx = 0; idx < nLabels; idx++) {
            if (stats.at<int>(idx, CC_STAT_TOP) + stats.at<int>(idx, CC_STAT_HEIGHT)/2 < 1000) {
                int blob_area = stats.at<int>(idx, CC_STAT_AREA);
                if (blob_area > 5000 && blob_area > maxArea && blob_area < 15000) {
                    maxArea = blob_area;
                    maxIndex = idx;
                }
            }
        }
        //cout << "Max area: " << maxArea << endl;

        // create rectangle for blob
        if (maxIndex > -1) {
            cur_rect.width = stats.at<int>(maxIndex, CC_STAT_WIDTH);
            cur_rect.height = stats.at<int>(maxIndex, CC_STAT_HEIGHT);
            cur_rect.x = stats.at<int>(maxIndex, CC_STAT_LEFT);
            cur_rect.y = stats.at<int>(maxIndex, CC_STAT_TOP);

            // ensure that this rectangle doesn't intersect with any previous ones
            if (!countNonZero(previous_rects(cur_rect))) {
                previous_rects(cur_rect).setTo(255);
            }



            // initialize if not already done so
            if (prev_rect.width != -1) {
                // check if cur_rect does not intersect with prev_rect
                if ((prev_rect & cur_rect).area() == 0) {
                    // if no intersection, stamp prev_rect
                    prev_rect = cur_rect;
                    frame(prev_rect).copyTo(background(prev_rect));
                }
            } else {
                // first good rectangle, so stamp it
                prev_rect = cur_rect;
                frame(prev_rect).copyTo(background(prev_rect));
            }

            rectangle(frame, cur_rect, Scalar(0, 0, 255));
        }
        // display output
        imshow(original_vid_window, frame);
        imshow(fg_mask_window, fgMaskMOG2);
        waitKey(1);
    }
    destroyAllWindows();

    // setup to display the mosiac
    string output_window = "Diver Mosaic (press q to continue)";
    namedWindow(output_window, WINDOW_FREERATIO);
    // resize to fit nicely on my screen, keeping aspect ratio the same
    resizeWindow(output_window, 360, 640);
    //moveWindow(output_window, 1700, 0);
    moveWindow(output_window, 0, 0);

    // get and display the background model
    cout << "Displaying diver mosaic (press q to continue)" << endl;
    imshow(output_window, background);

    string test = "test";
    namedWindow(test, WINDOW_FREERATIO);
    resizeWindow(test, 360, 640);
    moveWindow(test, 361, 0);
    imshow(test, previous_rects);

    // wait until user presses 'q' to move on
    while ((char)waitKey(0) != 'q') {}
    destroyAllWindows();
}