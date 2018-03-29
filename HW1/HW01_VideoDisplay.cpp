#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <dirent.h>

using namespace std;
using namespace cv;

/*
 * Video playing inspired by https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
 */
void playVid(VideoCapture vid) {
    // get fps of the video
    double fps = vid.get(CV_CAP_PROP_FPS);
    // figure out how long to wait between showing each frame of the video
    int incrementer = (int)(1000.0f/fps);
    // show frames for 3000ms, i.e. 3 seconds
    for(int counter = 0; counter <= 3000; counter+= incrementer) {
        Mat frame;
        // grab the next frame of the vid
        vid >> frame;
        // stop if we reach the end of the vid
        if (frame.empty()) {
            break;
        }
        // show the frame in a smallish window
        // change to namedWindow("Video", WINDOW_AUTOSIZE) for showing videos in their native resolution instead of scaled-down
        namedWindow("Video", WINDOW_GUI_EXPANDED);
        imshow("Video", frame);
        // wait between each frame
        waitKey(incrementer);
    }
    // cleanup
    destroyWindow("Video");
}


/*
 * Directory reading inspired by https://stackoverflow.com/a/612176/5496345
 */
int main(int argc, char *argv[]) {
    // get list of files in ./videos/
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir("./videos/"))) {
        while (ent = readdir(dir)) {
            // exclude . and ..
            if (strcmp(ent->d_name, ".\0") && strcmp(ent->d_name, "..\0")) {
                // create the filename string
                char *filename = (char *)malloc(sizeof(char) * (10 + strlen(ent->d_name)));
                filename = strncpy(filename, "./videos/\0", 10 + strlen(ent->d_name));
                filename = strcat(filename, ent->d_name);
                printf("Now attempting to play: \"%s\"\n", filename);
                // open the video file
                VideoCapture vid(filename);
                // Check if camera opened successfully
                if(!vid.isOpened()){
                    std::cout << "Error! Can't open video file: " << filename << std::endl;
                }
                else {
                    playVid(vid);
                    // cleanup
                    vid.release();
                }
                // free the earlier-malloced filename
                free(filename);
            }
        }
        closedir(dir);
    }
    else {
        // could not open directory
        std::cout << "Error! ./videos/ doesn't exist." << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}