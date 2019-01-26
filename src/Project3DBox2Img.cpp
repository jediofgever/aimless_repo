/**
 * @author Fetullah Atas
 * @email [fetullah.atas@mindtronicai.com]
 * @create date 2018-09-22 14:04:37
 * @modify date 2018-09-22 14:04:37
 * @desc [description]
 */

// Read KITTI calib , label , and images ,
// Projects 2D and 3D BB with use of P2 projection matrix
// Requires , opencv, Eigen and standart C++ LIBRARY

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

// Decleare paths for calib , label and images
int kNumberOfLabelFiles = 7481;
std::string label_dir_path = "/home/atas/MSc_Thesis/dataset/train/label_2/";
std::string calib_dir_path = "/home/atas/MSc_Thesis/dataset/train/calib/";
std::string image_dir_path = "/home/atas/MSc_Thesis/dataset/train/image_2/";

// extensions of files
std::string label_file_ext = ".txt";
std::string image_file_ext = ".png";

// Declare KITTI dateset label properties
// https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md

std::string truncated, occluded, alpha, score;
double rotation_yaw;
int Numof_cars = 0;

// input ; Label File;
// output ; vector container which has , BBOX, DIMENSIONS , POSITIONS in an
// order, yaw_rotation is also in last element os postion (position[3])
void GetAllCarInfoofFrame(
    std::ifstream& infile,
    std::vector<std::vector<std::vector<double> > >& All_Cars,
    std::vector<std::string>& Label_info) {
    std::string line;
    std::string type;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);

        iss >> type;

        if (type == "Car") {
            std::vector<std::vector<double> > BBBOX_dim_pos;

            std::vector<double> BBox(4, 0);
            std::vector<double> dimensions(3, 0);
            std::vector<double> position(4, 0);

            iss >> truncated >> occluded >> alpha >> BBox[0] >> BBox[1] >>
                BBox[2] >> BBox[3] >> dimensions[0] >> dimensions[1] >>
                dimensions[2] >> position[0] >> position[1] >> position[2] >>
                position[3] >> score;

            BBBOX_dim_pos.push_back(BBox);
            BBBOX_dim_pos.push_back(dimensions);
            BBBOX_dim_pos.push_back(position);
            All_Cars.push_back(BBBOX_dim_pos);
            Label_info.push_back(truncated);
            Label_info.push_back(occluded);
        }
    }
}

// Input : Read calibration file
// Output : P2 left camera projection and R0_Rect , Rectification matrix
void GetCalibofFrame(std::ifstream& infile, Eigen::Matrix4Xf& P2,
                     Eigen::Matrix4Xf& R0_rect) {
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "P2:") {
            // std::cout << type << endl;
            iss >> P2(0, 0) >> P2(0, 1) >> P2(0, 2) >> P2(0, 3) >> P2(1, 0) >>
                P2(1, 1) >> P2(1, 2) >> P2(1, 3) >> P2(2, 0) >> P2(2, 1) >>
                P2(2, 2) >> P2(2, 3);
            P2(3, 0) = P2(3, 1) = P2(3, 2) = 0.0;
            P2(3, 3) = 1.0;
        }

        if (type == "R0_rect:") {
            // std::cout << type << endl;
            iss >> R0_rect(0, 0) >> R0_rect(0, 1) >> R0_rect(0, 2) >>
                R0_rect(1, 0) >> R0_rect(1, 1) >> R0_rect(1, 2) >>
                R0_rect(2, 0) >> R0_rect(2, 1) >> R0_rect(2, 2);

            R0_rect(0, 3) = R0_rect(1, 3) = R0_rect(2, 3) = 0.0;
            R0_rect(3, 0) = R0_rect(3, 1) = R0_rect(3, 2) = 0.0;
            R0_rect(3, 3) = 1.0;
        }
    }
}

// This method Claculates 8 korners of box
// input ; Dimensions, positions of 3D BOX
// Output; 3D coordinates of each korner in camera frame
// Camera frame is defined as , x ; right , y ; down , z; forward
Eigen::MatrixXd ComputeCorners(std::vector<double> dimensions,
                               std::vector<double> positions) {
    double ry = positions[3];
    double h, w, l;
    h = dimensions[0];
    l = dimensions[1];
    w = dimensions[2];

    Eigen::MatrixXd corners(3, 8);
    Eigen::Matrix3d rot;
    rot << +cos(ry), 0, +sin(ry), 0, 1, 0, -sin(ry), 0, +cos(ry);

    Eigen::MatrixXd x(1, 8);
    x << w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2;

    Eigen::MatrixXd z(1, 8);
    z << -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2;

    Eigen::MatrixXd y(1, 8);
    y << 0, 0, 0, 0, -h, -h, -h, -h;

    corners.row(0) = x;
    corners.row(1) = y;
    corners.row(2) = z;

    corners = rot * corners;

    for (int k = 0; k < x.size(); k++) {
        corners(0, k) += positions[0];
        corners(1, k) += positions[1];
        corners(2, k) += positions[2];
    }
    return corners;
}

std::ofstream mtcnn_file;
void WriteLabelforMTCNN() {
    mtcnn_file.open("mtcnn_label.txt");
    mtcnn_file << "Writing this to a file.\n";
    mtcnn_file << "Writing this to a file.\n";
    mtcnn_file << "Writing this to a file.\n";
    mtcnn_file << "Writing this to a file.\n";
    mtcnn_file.close();
}

void ProcessProjection() {
    std::string space = " ";
    std::ofstream mtcnn_file, landmark_coordinate, gt_labels;
    //mtcnn_file.open("mtcnn_label.txt");
     //landmark_coordinate.open("landmark_coordinate.txt");
    gt_labels.open("train_bbx_gt.txt");

    for (int i = 0; i < kNumberOfLabelFiles; i++) {
        std::stringstream buffer;
        buffer << setfill('0') << setw(6) << i;
        std::string calib_file = calib_dir_path + buffer.str() + label_file_ext;
        std::string label_file = label_dir_path + buffer.str() + label_file_ext;
        std::string image_file = image_dir_path + buffer.str() + image_file_ext;

        std::ifstream label_infile(label_file.c_str());
        std::ifstream calib_infile(calib_file.c_str());

        cv::Scalar clr = cv::Scalar(0, 0, 255);
        cv::Scalar clr_b = cv::Scalar(255, 0, 0);
        cv::Scalar clr_ta = cv::Scalar(0, 255, 255);

        cv::Mat frame = cv::imread(image_file, 1);

        // Draw 12 lines that costructs box

        std::vector<std::vector<std::vector<double> > > All_Cars;
        std::vector<std::string> Label_info;

        GetAllCarInfoofFrame(label_infile, All_Cars, Label_info);

        Eigen::Matrix4Xf P2(4, 4);
        Eigen::Matrix4Xf R0_rect(4, 4);

        GetCalibofFrame(calib_infile, P2, R0_rect);
        if (All_Cars.size() > 0) {
             //mtcnn_file << image_file << space;
            gt_labels << image_file << "\n";
            gt_labels << All_Cars.size() << "\n";
        }

        for (int j = 0; j < All_Cars.size(); j++) {
            Eigen::MatrixXd corners =
                ComputeCorners(All_Cars[j][1], All_Cars[j][2]);

            std::vector<cv::Point> image_points;
            std::vector<cv::Point> landmark_points;

            for (int k = 0; k < 8; k++) {
                Eigen::MatrixXf corner_in_cam(4, 1);
                Eigen::Vector4f corner_in_image;

                corner_in_cam(0, 0) = corners(0, k);
                corner_in_cam(1, 0) = corners(1, k);
                corner_in_cam(2, 0) = corners(2, k);
                corner_in_cam(3, 0) = 1.0;

                corner_in_image = P2 * corner_in_cam;

                corner_in_image = corner_in_image / corner_in_image[2];

                cv::Point point;

                point.x = corner_in_image(0, 0);
                point.y = corner_in_image(1, 0);
                image_points.push_back(point);

                // Store korners in pixellls only of they are on image plane
                if (point.x >= 0 && point.x <= 1242) {
                    if (point.y >= 0 && point.y <= 375) {
                        landmark_points.push_back(point);
                    }
                }
            }

            cv::line(frame, image_points[0], image_points[1], clr_b, 1, 8);
            cv::line(frame, image_points[0], image_points[3], clr, 1, 8);
            cv::line(frame, image_points[0], image_points[4], clr_ta, 1, 8);
            cv::line(frame, image_points[1], image_points[2], clr, 1, 8);
            cv::line(frame, image_points[1], image_points[5], clr_ta, 1, 8);
            cv::line(frame, image_points[2], image_points[6], clr_ta, 1, 8);
            cv::line(frame, image_points[2], image_points[3], clr_b, 1, 8);
            cv::line(frame, image_points[3], image_points[7], clr_ta, 1, 8);
            cv::line(frame, image_points[7], image_points[4], clr, 1, 8);
            cv::line(frame, image_points[7], image_points[6], clr_b, 1, 8);
            cv::line(frame, image_points[4], image_points[5], clr_b, 1, 8);
            cv::line(frame, image_points[5], image_points[6], clr, 1, 8);

            /*namedWindow("Display window",
                        WINDOW_AUTOSIZE);     
            imshow("Display window", frame);  

            waitKey(0);*/

            cv::Point pt1, pt2;
            pt1.x = All_Cars[j][0][0];
            pt1.y = All_Cars[j][0][1];
            pt2.x = All_Cars[j][0][2];
            pt2.y = All_Cars[j][0][3];
            cv::Rect rect = cv::Rect(pt1, pt2);

            double width, height;
            width = pt2.x - pt1.x;
            height = pt2.y - pt1.y;

            gt_labels << pt1.x << space << pt1.y << space << width << space
                      << height << space << 1 << space << 0 << space << 0
                      << space << 0 << space << Label_info[1] << space << 0
                      << "\n";

            /*if (landmark_points.size() == 8) {
                landmark_coordinate
                    << image_file << space << pt1.x << space << pt1.y <<
            space
                    << pt2.x << space << pt2.y << space <<
            landmark_points[0].x
                    << space << landmark_points[0].y << space
                    << landmark_points[1].x << space << landmark_points[1].y
                    << space << landmark_points[3].x << space
                    << landmark_points[3].y << space << landmark_points[5].x
                    << space << landmark_points[5].y << space
                    << landmark_points[6].x << space << landmark_points[6].y
                    << "\n";
            }*/

            /*mtcnn_file << pt1.x << space << pt1.y << space << pt2.x <<
               space
                       << pt2.y << space;*/

            // cv::rectangle(frame, rect, clr, 2, 2, 0);

            cv::Mat car = cv::Mat(frame, rect);
            cv::Mat car_original = car;

            cvtColor(car, car, COLOR_RGB2GRAY);

            int thresh = 100;
            int max_thresh = 255;
            RNG rng(12345);
        }
        if (All_Cars.size() > 0) {
             mtcnn_file << "\n";
        }
        std::cout<<"finished "<<  i << std::endl;
    }

     //landmark_coordinate.close();
     gt_labels.close();
     //  mtcnn_file.close();
}

int main(int argc, char const* argv[]) {
    std::cout << "Hello World" << std::endl;
    ProcessProjection();

    // WriteLabelforMTCNN();

    return 0;
}
