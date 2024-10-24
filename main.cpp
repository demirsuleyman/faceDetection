#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



using namespace cv;
using namespace std;

int main()
{
    //Load the einstein.jpg
    Mat einstein = imread("einstein.jpg",0);
    if (einstein.empty())
    {
        cout<<"Einstein not found"<<endl;
        return -1;
    }

    //Load cascade classifier for face detection
    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml"))
    {
        cout<<"Load face cascade failed"<<endl;
        return -1;
    }

    //Face detection on einstein.jpg
    vector<Rect> faces;
    face_cascade.detectMultiScale(einstein,faces);

    //Draw the rectangle on detected face
    for (const auto &face:faces)
    {
        rectangle(einstein,face,Scalar(255,255,255),10);
    }

    //Show the einstein.jpg
    namedWindow("Einstein",WINDOW_NORMAL);
    imshow("Einstein",einstein);

    // If the q key is pressed then close Einstein window.
    if (waitKey(0) == 'q'){destroyWindow("Einstein");}



    //Load the barcelona.jpg
    Mat barcelona = imread("barcelona.jpg",0);
    if(barcelona.empty())
    {
        cout<<"Barcelona not found"<<endl;
        return -1;
    }

    //Face detection on barcelona.jpg
    vector<Rect> faces_barcelona;
    face_cascade.detectMultiScale(barcelona,faces_barcelona,1.1,7);

    // Draw the rectangle on detected face
    for (const auto &face:faces_barcelona)
    {
        rectangle(barcelona,face,Scalar(255,255,255),10);
    }

    // Show the barcelona.jpg
    namedWindow("Barcelona",WINDOW_NORMAL);
    imshow("Barcelona",barcelona);
    // If the q key is pressed then close the Barcelona window.
    if (waitKey(0) == 'q'){destroyWindow("Barcelona");}


    //Video capture
    VideoCapture cap = VideoCapture(0);
    if(!cap.isOpened())
    {
        cout<<"Video capture failed"<<endl;
        return -1;
    }

    Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        // Face detection on frame
        vector<Rect> faces_video;
        face_cascade.detectMultiScale(frame,faces_video,1.1,7);

        // Draw the rectangles on detected faces
        for (const auto &face:faces_video)
        {
            rectangle(frame,face,Scalar(255,255,255),10);
        }
        // Show the frame
        imshow("Face Detection",frame);

        // If the q key is pressed then break the while loop.
        if (waitKey(1) == 'q') break;
    }

    // Release resources
    cap.release();
    destroyAllWindows();
    return 0;

}
