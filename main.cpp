#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



using namespace cv;
using namespace std;

int main()
{
    Mat einstein = imread("/home/demir/CLionProjects/faceDetection/einstein.jpg",0);
    if (einstein.empty())
    {
        cout<<"Einstein not found"<<endl;
        return -1;
    }

    CascadeClassifier face_cascade;
    if (!face_cascade.load("/home/demir/CLionProjects/faceDetection/haarcascade_frontalface_default.xml"))
    {
        cout<<"Load face cascade failed"<<endl;
        return -1;
    }

    vector<Rect> faces;
    face_cascade.detectMultiScale(einstein,faces);

    for (const auto &face:faces)
    {
        rectangle(einstein,face,Scalar(255,255,255),10);
    }

    namedWindow("Einstein",WINDOW_NORMAL);
    imshow("Einstein",einstein);
    waitKey(0);


    Mat barcelona = imread("/home/demir/CLionProjects/faceDetection/barcelona.jpg",0);
    if(barcelona.empty())
    {
        cout<<"Barcelona not found"<<endl;
        return -1;
    }

    vector<Rect> faces_barcelona;
    face_cascade.detectMultiScale(barcelona,faces_barcelona,1.1,7);
    for (const auto &face:faces_barcelona)
    {
        rectangle(barcelona,face,Scalar(255,255,255),10);
    }

    namedWindow("Barcelona",WINDOW_NORMAL);
    imshow("Barcelona",barcelona);
    waitKey(0);


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
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> faces_video;
        face_cascade.detectMultiScale(gray,faces_video,1.1,7);

        for (const auto &face:faces_video)
        {
            rectangle(gray,face,Scalar(255,255,255),10);
        }
        imshow("Face Detection",gray);

        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;

}
