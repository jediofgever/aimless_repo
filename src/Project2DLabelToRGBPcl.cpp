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
std::string image_dir_path = "/home/atas/MSc_Thesis/dataset/train/rgb_pcl/";

std::string calib_file_path_V2C =
    "/home/atas/MSc_Thesis/dataset/2011_09_26/calib_velo_to_cam.txt";

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

void GetTCamVelo(std::ifstream& infile, Eigen::Matrix4Xf& in_matrix) {
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "R:") {
            iss >> in_matrix(0, 0) >> in_matrix(0, 1) >> in_matrix(0, 2) >>
                in_matrix(1, 0) >> in_matrix(1, 1) >> in_matrix(1, 2) >>
                in_matrix(2, 0) >> in_matrix(2, 1) >> in_matrix(2, 2);
        }
        if (type == "T:") {
            iss >> in_matrix(0, 3) >> in_matrix(1, 3) >> in_matrix(2, 3);
        }

        in_matrix(3, 0) = in_matrix(3, 1) = in_matrix(3, 2) = 0.0;
        in_matrix(3, 3) = 1.0;
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

void ProcessProjection() {
    Eigen::Matrix4Xf T_cam_velo(4, 4);
    std::ifstream calib_file_path_infile(calib_file_path_V2C.c_str());
    GetTCamVelo(calib_file_path_infile, T_cam_velo);

    for (int i = 0; i < kNumberOfLabelFiles; i++) {
        std::cout << i << std::endl;

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

        namedWindow("Display window",
                    WINDOW_AUTOSIZE);     // Create a window for display.
        imshow("Display window", frame);  // Show our image inside it.

        waitKey(0);

        std::vector<std::vector<std::vector<double> > > All_Cars;
        std::vector<std::string> Label_info;

        GetAllCarInfoofFrame(label_infile, All_Cars, Label_info);

        for (int j = 0; j < All_Cars.size(); j++) {
            Eigen::MatrixXd corners =
                ComputeCorners(All_Cars[j][1], All_Cars[j][2]);

            cv::Point pt1_on2D, pt2_on2D, pt3_on2D, pt4_on2D;

            pt1_on2D.y = corners(2, 4);
            pt1_on2D.x = corners(0, 4);

            pt2_on2D.y = corners(2, 5);
            pt2_on2D.x = corners(0, 5);

            pt3_on2D.y = corners(2, 6);
            pt3_on2D.x = corners(0, 6);

            pt4_on2D.y = corners(2, 7);
            pt4_on2D.x = corners(0, 7);

            pt1_on2D.y -= 60;
            pt1_on2D.x += 15;

            pt2_on2D.y -= 60;
            pt2_on2D.x += 15;

            pt3_on2D.y -= 60;
            pt3_on2D.x += 15;

            pt4_on2D.y -= 60;
            pt4_on2D.x += 15;

            // scale up to image dimensions 600 x 450
            pt1_on2D.x *= 15;
            pt2_on2D.x *= 15;

            pt1_on2D.y *= -10;
            pt2_on2D.y *= -10;

            pt3_on2D.x *= 15;
            pt4_on2D.x *= 15;

            pt3_on2D.y *= -10;
            pt4_on2D.y *= -10;

            cv::Rect rect_on2D = cv::Rect(pt1_on2D, pt2_on2D);

            if (pt1_on2D.x > 0 && pt1_on2D.x < 450 && pt2_on2D.x > 0 &&
                pt2_on2D.x < 450) {
                if (pt1_on2D.y > 0 && pt1_on2D.y < 600 && pt2_on2D.y > 0 &&
                    pt2_on2D.y < 600) {
                    // cv::rectangle(frame, rect_on2D, c, 2, 2, 0);

                    cv::line(frame, pt1_on2D, pt2_on2D, clr, 1, 8);
                    cv::line(frame, pt2_on2D, pt3_on2D, clr, 1, 8);
                    cv::line(frame, pt3_on2D, pt4_on2D, clr, 1, 8);
                    cv::line(frame, pt4_on2D, pt1_on2D, clr, 1, 8);
                }
            }

            image_file =
                image_dir_path + buffer.str() + buffer.str() + image_file_ext;
            cv::imwrite(image_file, frame);
            std::cout << "wrote image" << std::endl;
        }
    }
}

int main(int argc, char const* argv[]) {
    std::cout << "Hello World" << std::endl;

    ProcessProjection();

    // WriteLabelforMTCNN();

    return 0;
}
