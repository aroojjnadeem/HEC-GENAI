#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <mpi.h>

using namespace cv;
using namespace cv::face;
using namespace std;

#define MAX_IMAGE 200000

int my_rank, p;
MPI_Comm comm;

String face_cascade_name = "cascades/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

std::vector<Rect> detectFaces(Mat frame_gray)
{
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE);
    return faces;
}

static void read_csv(const string &filename, vector<Mat> &images, vector<int> &labels, char separator = ';')
{

    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file)
    {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }

    string line, path, classlabel;
    int position;
    Mat A, B;
    char outbuf[MAX_IMAGE];
    unsigned char *data;
    int rows, cols, label, i = 0;

    while (getline(file, line))
    {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty())
        {
            position = 0;
            label = atoi(classlabel.c_str());
            if (my_rank == 0)
            {
                A = imread(path, IMREAD_GRAYSCALE);
                B = Mat(A.rows, A.cols, 0, A.data);
                if (label % p == 0)
                {
                    imwrite(format("c0/testemean%i.png", i++), B);
                    images.push_back(A);
                    labels.push_back(label);
                    continue;
                }

                MPI_Pack(&A.rows, 1, MPI_INT, &outbuf, MAX_IMAGE, &position, comm);
                MPI_Pack(&A.cols, 1, MPI_INT, &outbuf, MAX_IMAGE, &position, comm);
                MPI_Pack(A.data, A.rows * A.cols, MPI_CHAR, &outbuf, MAX_IMAGE, &position, comm);
                MPI_Send(&outbuf, MAX_IMAGE, MPI_PACKED, label % p, 0, comm);
            }
            else
            {
                if (label % p != my_rank)
                    continue;

                MPI_Recv(&outbuf, MAX_IMAGE, MPI_PACKED, 0, 0, comm, MPI_STATUS_IGNORE);

                MPI_Unpack(&outbuf, MAX_IMAGE, &position, &rows, 1, MPI_INT, comm);
                MPI_Unpack(&outbuf, MAX_IMAGE, &position, &cols, 1, MPI_INT, comm);

                data = (unsigned char *)malloc(rows * cols * sizeof(unsigned char));
                MPI_Unpack(&outbuf, MAX_IMAGE, &position, data, rows * cols, MPI_CHAR, comm);

                A = Mat(rows, cols, 0, data);
                images.push_back(A);
                labels.push_back(label);
                imwrite(format("c1/testemean%i.png", i++), A);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cout << "usage: " << argv[0] << " <csv.ext> <input image>" << endl;
        exit(1);
    }

    if (!face_cascade.load(face_cascade_name))
    {
        printf("--(!)Error loading face cascade\n");
        return -1;
    }

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    string fn_csv = string(argv[1]);
    vector<Mat> images;
    vector<int> labels;

    try
    {
        read_csv(fn_csv, images, labels);
    }
    catch (cv::Exception &e)
    {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    if (images.size() <= 1)
    {
        CV_Error(Error::StsError, "This demo needs at least 2 images to work.");
    }

    Mat frame, frame_gray, testSample;
    std::vector<Rect> faces;
    int position, rows, cols;
    double start, finish, elapsed, local_elapsed;
    char inputbuf[MAX_IMAGE];
    unsigned char *inputdata;
    int predictedLabel;
    double local_confidence = 0.0, confidence = 0.0, predictImage;

    frame = imread("crowd.jpg");
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    faces = detectFaces(frame_gray);

    MPI_Barrier(comm);

    Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create();
    model->train(images, labels);

    start = MPI_Wtime();

    for (int i = 0; i < faces.size(); i++)
    {
        position = 0;

        if (my_rank == 0)
        {
            testSample = frame_gray(faces[i]);
            MPI_Pack(&testSample.rows, 1, MPI_INT, &inputbuf, MAX_IMAGE, &position, comm);
            MPI_Pack(&testSample.cols, 1, MPI_INT, &inputbuf, MAX_IMAGE, &position, comm);
            MPI_Pack(testSample.data, testSample.rows * testSample.cols, MPI_CHAR, &inputbuf, MAX_IMAGE, &position, comm);
            MPI_Bcast(&inputbuf, MAX_IMAGE, MPI_PACKED, 0, comm);
        }
        else
        {
            MPI_Bcast(&inputbuf, MAX_IMAGE, MPI_PACKED, 0, comm);
            MPI_Unpack(&inputbuf, MAX_IMAGE, &position, &rows, 1, MPI_INT, comm);
            MPI_Unpack(&inputbuf, MAX_IMAGE, &position, &cols, 1, MPI_INT, comm);
            inputdata = (unsigned char *)malloc(rows * cols * sizeof(unsigned char));
            MPI_Unpack(&inputbuf, MAX_IMAGE, &position, inputdata, rows * cols, MPI_CHAR, comm);
            testSample = Mat(rows, cols, 0, inputdata);
        }

        model->predict(testSample, predictedLabel, local_confidence);
        MPI_Barrier(comm);

        MPI_Allreduce(&local_confidence, &confidence, 1, MPI_DOUBLE, MPI_MIN, comm);

        if (local_confidence == confidence && my_rank == 0)
            predictImage = predictedLabel;
        if (local_confidence == confidence && my_rank != 0)
            MPI_Send(&predictedLabel, 1, MPI_INT, 0, 0, comm);
        else if (local_confidence != confidence && my_rank == 0)
            MPI_Recv(&predictImage, 1, MPI_INT, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);

        free(inputdata);

        if (my_rank == 0)
        {
            Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
            printf("confidence %f \t Id %i\n", confidence, (int)predictImage);
            if (confidence < 120)
                rectangle(frame, faces[i], Scalar(0, 0, 255), 2);
            else
                rectangle(frame, faces[i], Scalar(0, 255, 0), 2);
        }
    }

    finish = MPI_Wtime();
    local_elapsed = finish - start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (my_rank == 0)
    {
        printf("Elapsed time %f\n", elapsed);
        imwrite("final.jpg", frame);
    }

    MPI_Finalize();
    return 0;
}
