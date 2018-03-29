#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <dirent.h>
#include <cmath>

using namespace cv;
using namespace std;

/**
 * Returns average frame brightness or lightness, out of 255
 * @param bgrFrame Input matrix as an array of BGR scalars
 * @return average frame brightness or lightness, out of 255
 */
double avgLightnessForFrame(Mat bgrFrame) {
    Mat hsvFrame;
    cvtColor(bgrFrame, hsvFrame, COLOR_BGR2GRAY);

    Scalar avgHSV = mean(hsvFrame);
    return avgHSV[0];
}


//Water level is pixels top to level of water
void mosaicVideoAtPath(char *filepath, char *window, int learningFrames, int waterLevel) {
    VideoCapture curCapture(filepath);

    if (curCapture.isOpened()) {
        cout << "Attempting to play " << filepath << " as video.\n";

        int fps = curCapture.get(CV_CAP_PROP_FPS);

        cout << "Playing at " << fps << " fps, learning for " << learningFrames << " frames.\n";

        Mat curFrame;

        Mat mosaic;

        int minBrightness = 255;
        int maxBrightness = 0;
        int frameBrightness;

        int maxDiverArea = 8000;//px
        int minDiverArea = 5000;//px

        int frameOffset = 0;

        for (int frame = 0; frame < learningFrames; frame++) { //Learning
            curCapture >> curFrame;
            if (!curFrame.empty()) {
                frameBrightness = avgLightnessForFrame(curFrame);

                if (frameBrightness < minBrightness) {
                    minBrightness = frameBrightness;
                } else if (frameBrightness > maxBrightness) {
                    maxBrightness = frameBrightness;
                }

                imshow(window, curFrame);
            }

            if (waitKey(fps) >= 0) {
                return; //Skip to next video
            }
            frameOffset++;
        }

        mosaic = curFrame.clone();

        int avgBrightness = (minBrightness + maxBrightness) / 2;
        cout << "Avg brightness: " << avgBrightness << "\n";
        cout << "Continuing, removing low frames\n";

        bool relearnedAvg = false;

        int moreThanAvg = (avgBrightness + ((maxBrightness - avgBrightness) / 3));

        int firstFlickerFrame = -1; //TODO
        int secondFlickerFrame = -1;
        int flickerSpacing = -1;

        fps = (int)lround(curCapture.get(CV_CAP_PROP_FPS));

        cout << "Currently at " << fps << " fps\n";

        Ptr<BackgroundSubtractor> pBGSubtractor; //MOG2 background subtractor
        pBGSubtractor = createBackgroundSubtractorMOG2(40, 45, false);
        Mat bgMask(curFrame.size(), CV_8U);
        Mat adjustedMask(curFrame.size(), CV_8U);
        Mat diverMask(curFrame.size(), CV_8U);
        Mat out(curFrame.size(), CV_8UC3);

        /*
        vector<Point> diverPoints;
        vector<vector<Point>> contours;
        vector<Vec4i> contourHierarchy;
        */

        Mat ccLabels, ccStats, ccCentroids;

        Rect prevDiverRect(0, 0, 0, 0);

        int kernelWidth = 3;
        int kernelWidth2 = 8;
        Mat kernel = getStructuringElement(
                MORPH_ELLIPSE, Size(kernelWidth, kernelWidth),
                Point(kernelWidth / 2, kernelWidth / 2)
        );

        Mat kernel2 = getStructuringElement(
                MORPH_ELLIPSE, Size(kernelWidth2, kernelWidth2),
                Point(kernelWidth2 / 2, kernelWidth2 / 2)
        );

        for (;;) { //Second learning phase
            curCapture >> curFrame;
            if (!curFrame.empty()) {
                int frameBrightness = avgLightnessForFrame(curFrame);

                if (frameBrightness > maxBrightness) {
                    //cout << "Relearning max value!\n";
                    maxBrightness = frameBrightness;
                    relearnedAvg = true;
                } else if (frameBrightness < minBrightness) {
                    //cout << "Relearning min value!\n";
                    minBrightness = frameBrightness;
                    relearnedAvg = true;
                }

                if (relearnedAvg) {
                    avgBrightness = (minBrightness + maxBrightness) / 2;
                    moreThanAvg = (avgBrightness + ((maxBrightness - avgBrightness) / 3));
                    relearnedAvg = false;
                }

                if (frameBrightness < moreThanAvg) {
                    continue;
                }

                pBGSubtractor->apply(curFrame, bgMask);

                //Fill holes
                //morphologyEx(bgMask, adjustedMask, MORPH_CLOSE, kernel );

                erode(bgMask, adjustedMask, kernel);

                //Remove noise
                //morphologyEx(adjustedMask, adjustedMask, MORPH_OPEN, kernel/2 );

                //Merge blobs
                //dilate(adjustedMask, adjustedMask, kernel, Point(-1,-1), 2);

                //Remove noise again
                //morphologyEx(adjustedMask, adjustedMask, MORPH_CLOSE, kernel );

                threshold(adjustedMask, adjustedMask, 1, 255, THRESH_BINARY);

                //findContours(adjustedMask, contours, contourHierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
                /*
                for (int contour=0; contour<numContours; contour++) {
                    double area = contourArea(contours[contour]);
                    if (area > maxArea) {
                        maxArea = area;
                        cIndex = contour;
                    }
                }

                if (cIndex >= 0 && maxArea > 40 && maxArea <= 400) {
                    convexHull(contours[cIndex], diverPoints);

                    //cout << maxArea << "\n";

                    fillConvexPoly(diverMask, diverPoints, 255);
                }*/

                int numCCs = connectedComponentsWithStats(adjustedMask, ccLabels, ccStats, ccCentroids, 8);
                //cout << numCCs << " ccs\n";

                int maxArea = 0, cIndex = -1;
                Rect ccRect;


                for (int cc = 1; cc < numCCs; cc++) {
                    int ccArea = ccStats.at<int>(cc, CC_STAT_AREA);
                    if (
                            ccArea > maxArea &&
                            ccArea <= maxDiverArea &&
                            ccArea >= minDiverArea
                            ) {
                        //cout << ccArea << "\n";
                        maxArea = ccArea;
                        cIndex = cc;

                        ccRect.width = ccStats.at<int>(cc, CC_STAT_WIDTH);
                        ccRect.height = ccStats.at<int>(cc, CC_STAT_HEIGHT);
                        ccRect.x = ccStats.at<int>(cc, CC_STAT_LEFT);
                        ccRect.y = ccStats.at<int>(cc, CC_STAT_TOP);
                    }
                }

                //cout << maxArea << " is max area. " << cIndex << " is idx \n";

                if (cIndex > 0) { //Is not unknown CC
                    //Check if last diver spot overlaps this one
                    bool rectsCollide = (prevDiverRect & ccRect).area() > 0 ||
                                        (ccRect.y > 900); //Ignore water
                    if (!rectsCollide) {
                        for (int row = ccRect.y; row < ccRect.height + ccRect.y; row++) {
                            for (int col = ccRect.x; col < ccRect.width + ccRect.x; col++) {
                                bool pixelInBGMask = bgMask.at<int>(row, col) != 0;
                                int label = ccLabels.at<int>(row, col);
                                //cout << pixel << "\n";
                                //out.at<Vec3b>(row, col) = pixelToCheck ? curFrame.at<Vec3b>(row, col) : 0;
                                if (label == cIndex || pixelInBGMask) {
                                    mosaic.at<Vec3b>(row, col) = curFrame.at<Vec3b>(row, col);
                                }
                                // mosaic.at<Vec3b>(row, col) = (label == cIndex && pixelToShow) ? curFrame.at<Vec3b>(row, col);
                            }
                        }

                        //rectangle(mosaic, ccRect, Scalar(255, 0, 0), 1);

                        prevDiverRect = ccRect;
                        imshow(window, mosaic);
                    }
                }

                /*if (cIndex > 0) {
                    circle(adjustedMask,
                        Point(ccCentroids.at<double>(cIndex, 0), ccCentroids.at<double>(cIndex, 1)),
                        30, Scalar(128,0,0), -1
                    );
                }*/

                imshow(window, curFrame);


            } else {
                return;
            }
            if (waitKey(fps) >= 0) {
                return; //Skip to next video
                //waitKey(0);
            }
            frameOffset++;
        }

    } else {
        cout << "ERROR: Couldn't open file at " << filepath << " as video.\n";
    }
}

int main(int argc, char **argv) {
    char *vidFolderLoc = "/home/mark/Documents/College/Advanced_Computer_Vision/HWs/HW03_Philip_Mark/videos";
    char *window = "Video Display";

    if (argc == 2) {
        vidFolderLoc = argv[1];
    }

    cout << "Loading videos in folder at: " << vidFolderLoc << "\n";

    namedWindow(window, WINDOW_NORMAL); //create window

    DIR *dirPointer;
    int i;
    string fileNameAndPath = "";
    string fileName = "";
    char *filePathOutPointer = "";
    struct dirent *dir;
    dirPointer = opendir(vidFolderLoc);
    if (dirPointer != NULL) {
        while ((dir = readdir(dirPointer)) != NULL) {
            fileName = dir->d_name;
            fileNameAndPath = vidFolderLoc + string("/") + fileName;
            if (fileName[0] == '.') {
                continue;
            }

            filePathOutPointer = &fileNameAndPath[0];

            cout << "Found file: " << filePathOutPointer << "\n";

            mosaicVideoAtPath(filePathOutPointer, window, 12, 800);
        }
        closedir(dirPointer);
    } else {
        cout << "ERROR: Couldn't open given directory: " << vidFolderLoc << "\n";
    }


    return 0;
}