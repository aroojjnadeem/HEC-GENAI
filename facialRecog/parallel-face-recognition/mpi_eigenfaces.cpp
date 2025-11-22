/*
 * Updated for OpenCV 4.12.0 by ChatGPT (GPT-5)
 * Original: Philipp Wagner (BSD License)
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp> // for CascadeClassifier (optional)
#include <opencv2/face.hpp>      // from opencv_contrib
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <mpi.h>

using namespace cv;
using namespace std;
using namespace cv::face;

#define MAX_IMAGE 200000

int my_rank, p;
MPI_Comm comm;

static void read_csv(const string &filename, vector<Mat> &images, vector<int> &labels, char separator = ';')
{
    ifstream file(filename.c_str(), ifstream::in);
    if (!file.is_open())
    {
        cerr << "Error: Cannot open CSV file " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string line, path, classlabel;
    int position;
    Mat A;
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
            label = stoi(classlabel);

            if (my_rank == 0)
            {
                A = imread(path, IMREAD_GRAYSCALE);
                if (A.empty())
                    continue;

                if (label % p == 0)
                {
                    imwrite(format("c0/testemean%d.png", i++), A);
                    images.push_back(A);
                    labels.push_back(label);
                    continue;
                }

                MPI_Pack(&A.rows, 1, MPI_INT, outbuf, MAX_IMAGE, &position, comm);
                MPI_Pack(&A.cols, 1, MPI_INT, outbuf, MAX_IMAGE, &position, comm);
                MPI_Pack(A.data, A.rows * A.cols, MPI_UNSIGNED_CHAR, outbuf, MAX_IMAGE, &position, comm);

                MPI_Send(outbuf, MAX_IMAGE, MPI_PACKED, label % p, 0, comm);
            }
            else
            {
                if (label % p != my_rank)
                    continue;

                MPI_Recv(outbuf, MAX_IMAGE, MPI_PACKED, 0, 0, comm, MPI_STATUS_IGNORE);

                MPI_Unpack(outbuf, MAX_IMAGE, &position, &rows, 1, MPI_INT, comm);
                MPI_Unpack(outbuf, MAX_IMAGE, &position, &cols, 1, MPI_INT, comm);

                data = (unsigned char *)malloc(rows * cols);
                MPI_Unpack(outbuf, MAX_IMAGE, &position, data, rows * cols, MPI_UNSIGNED_CHAR, comm);

                A = Mat(rows, cols, CV_8UC1, data);
                images.push_back(A);
                labels.push_back(label);
                imwrite(format("c1/testemean%d.png", i++), A);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cout << "Usage: " << argv[0] << " <csv.ext> <test_image>" << endl;
        return 1;
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
        cerr << "Error reading CSV: " << e.msg << endl;
        MPI_Abort(comm, 1);
    }

    if (images.size() <= 1)
    {
        cerr << "Need at least 2 images." << endl;
        MPI_Abort(comm, 1);
    }

    Mat testSample;
    char inputbuf[MAX_IMAGE];
    unsigned char *inputdata = nullptr;
    int rows, cols, position = 0;
    double start, finish, elapsed, local_elapsed;
    int predictedLabel = -1, predictImage = -1;
    double local_confidence = 0.0, confidence = 0.0;

    MPI_Barrier(comm);
    start = MPI_Wtime();

    if (my_rank == 0)
    {
        testSample = imread(argv[2], IMREAD_GRAYSCALE);
        if (testSample.empty())
        {
            cerr << "Error: Cannot read test image." << endl;
            MPI_Abort(comm, 1);
        }

        MPI_Pack(&testSample.rows, 1, MPI_INT, inputbuf, MAX_IMAGE, &position, comm);
        MPI_Pack(&testSample.cols, 1, MPI_INT, inputbuf, MAX_IMAGE, &position, comm);
        MPI_Pack(testSample.data, testSample.rows * testSample.cols, MPI_UNSIGNED_CHAR, inputbuf, MAX_IMAGE, &position, comm);
    }

    MPI_Bcast(inputbuf, MAX_IMAGE, MPI_PACKED, 0, comm);

    if (my_rank != 0)
    {
        MPI_Unpack(inputbuf, MAX_IMAGE, &position, &rows, 1, MPI_INT, comm);
        MPI_Unpack(inputbuf, MAX_IMAGE, &position, &cols, 1, MPI_INT, comm);

        inputdata = (unsigned char *)malloc(rows * cols);
        MPI_Unpack(inputbuf, MAX_IMAGE, &position, inputdata, rows * cols, MPI_UNSIGNED_CHAR, comm);
        testSample = Mat(rows, cols, CV_8UC1, inputdata);
    }

    // ✅ OpenCV 4 face module API
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    model->train(images, labels);
    model->predict(testSample, predictedLabel, local_confidence);

    MPI_Allreduce(&local_confidence, &confidence, 1, MPI_DOUBLE, MPI_MIN, comm);

    if (local_confidence == confidence && my_rank == 0)
        predictImage = predictedLabel;

    if (local_confidence == confidence && my_rank != 0)
        MPI_Send(&predictedLabel, 1, MPI_INT, 0, 0, comm);
    else if (local_confidence != confidence && my_rank == 0)
        MPI_Recv(&predictImage, 1, MPI_INT, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);

    finish = MPI_Wtime();
    local_elapsed = finish - start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (inputdata)
        free(inputdata);

    if (my_rank == 0)
    {
#ifndef SHOW_ONLY_TIME
        cout << "Cores: " << p << endl;
        cout << format("Predicted class = %d / confidence = %f\nElapsed time = %f s", predictImage, confidence, elapsed) << endl;
#else
        cout << elapsed << endl;
#endif

#ifdef DISPLAY
        imshow("Test Image", testSample);
        waitKey(0);
#endif
    }

    MPI_Finalize();
    return 0;
}
