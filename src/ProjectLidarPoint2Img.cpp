/**
 * @author Fetullah Atas
 * @email [fetullah.atas@mindtronicai.com]
 * @create date 2018-10-16 15:06:23
 * @modify date 2018-10-16 15:06:23
 * @desc [description]
 */
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Decleare paths for calib , label and images
int kNumberOfLabelFiles = 7480;

std::string calib_dir_path = "/home/atas/MSc_Thesis/dataset/train/calib/";
std::string image_dir_path =
    "/home/atas/MSc_Thesis/dataset/2011_09_26/2011_09_26_drive_0084_sync/"
    "image_02/data";

std::string rgb_pcl_image_dir_path =
    "/home/atas/MSc_Thesis/dataset/train/"
    "rgb_pcl/";

std::string velodyne_path =
    "/home/atas/MSc_Thesis/dataset/2011_09_26/2011_09_26_drive_0084_sync/"
    "velodyne_points/data"
    "/pcds/";

std::string calib_file_path_V2C =
    "/home/atas/MSc_Thesis/dataset/2011_09_26/calib_velo_to_cam.txt";

// extensions of files
std::string calib_file_ext = ".txt";
std::string label_file_ext = ".txt";
std::string image_file_ext = ".png";
std::string velodyne_file_ext = ".pcd";

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

void CreateRGB2DImageFromColorfulPCL(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud, std::string image_file) {
    cv::Mat bird_view_image(600, 450, CV_8UC3);
    bird_view_image.setTo(cv::Scalar(0, 0, 0));

    for (int r = 0; r < out_cloud->points.size(); r++) {
        if (out_cloud->points[r].z < 60 && out_cloud->points[r].x > -15 &&
            out_cloud->points[r].x < 15) {
            cv::Point point;

            out_cloud->points[r].z -= 60;
            out_cloud->points[r].x += 15;

            point.x = out_cloud->points[r].x * 15;
            point.y = -out_cloud->points[r].z * 10;

            cv::Vec3b rgb_pixel, dense;

            rgb_pixel[2] = out_cloud->points[r].r;
            rgb_pixel[1] = out_cloud->points[r].g;
            rgb_pixel[0] = out_cloud->points[r].b;

            dense[2] = 255;
            dense[1] = 0;
            dense[0] = 0;

            if (point.x > 0 && point.x < 450) {
                if (point.y > 0 && point.y < 600) {
                    bird_view_image.at<cv::Vec3b>(point.y, point.x) = rgb_pixel;
                }
            }
        }
    }

    cv::imwrite(image_file, bird_view_image);

    // pcl::io::savePCDFileASCII("test_pcd.pcd", *out_cloud);
}

void ProcessProjection() {
    std::string space = " ";

    Eigen::Matrix4Xf T_cam_velo(4, 4);
    std::ifstream calib_file_path_infile(calib_file_path_V2C.c_str());
    GetTCamVelo(calib_file_path_infile, T_cam_velo);

    for (int i = 0; i < kNumberOfLabelFiles; i++) {
        std::stringstream buffer;
        buffer << setfill('0') << setw(6) << i;
        std::string image_file = image_dir_path + buffer.str() + image_file_ext;
        std::string velodyne_file =
            velodyne_path + buffer.str() + velodyne_file_ext;

        std::string calib_file_path =
            calib_dir_path + buffer.str() + label_file_ext;

        std::ifstream calib_infile(calib_file_path.c_str());
        Eigen::Matrix4Xf P2(4, 4);
        Eigen::Matrix4Xf R0_rect(4, 4);
        GetCalibofFrame(calib_infile, P2, R0_rect);

        cv::Scalar clr_b = cv::Scalar(255, 0, 0);
        cv::Scalar clr_ta = cv::Scalar(0, 255, 255);

        cv::Mat frame = cv::imread(image_file, 1);

        std::vector<cv::Point> image_points;
        std::vector<cv::Point> landmark_points;

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZRGBA>);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>);

        out_cloud->is_dense = true;

        if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(velodyne_file.c_str(),
                                                    *cloud) ==
            -1)  //* load the file
        {
            PCL_ERROR("Couldn't read file test_pcd.pcd \n");
            return;
        }

        for (int k = 0; k < cloud->points.size(); k++) {
            Eigen::MatrixXf point_in_velo(4, 1);
            Eigen::Vector4f corner_in_image;
            Eigen::Vector4f corner_in_cam_3d;

            if (cloud->points[k].x > 0 /*&& cloud->points[k].z > -1.5*/) {
                // std::cout << cloud->points[k].z << std::endl;
                point_in_velo(0, 0) = cloud->points[k].x;
                point_in_velo(1, 0) = cloud->points[k].y;
                point_in_velo(2, 0) = cloud->points[k].z;
                point_in_velo(3, 0) = 1.0;
                double distance = sqrt(pow(cloud->points[k].x, 2) +
                                       pow(cloud->points[k].y, 2) +
                                       pow(cloud->points[k].z, 2));

                cv::Scalar clr = cv::Scalar(130, 120, distance * 15);

                // std::cout << point_in_velo << endl;

                corner_in_image = P2 * R0_rect * T_cam_velo * point_in_velo;

                corner_in_cam_3d = T_cam_velo * point_in_velo;

                corner_in_image = corner_in_image / corner_in_image[2];

                cv::Point point;

                point.x = corner_in_image(0, 0);
                point.y = corner_in_image(1, 0);

                // Store korners in pixellls only of they are on image plane
                if (point.x >= 0 && point.x <= 1242) {
                    if (point.y >= 0 && point.y <= 375) {
                        // cv::circle(frame, point, 1, clr, 1);

                        pcl::PointXYZRGB colored_3d_point;

                        cv::Vec3b rgb_pixel =
                            frame.at<cv::Vec3b>(point.y, point.x);

                        colored_3d_point.x = corner_in_cam_3d(0, 0);
                        colored_3d_point.y = corner_in_cam_3d(1, 0);
                        colored_3d_point.z = corner_in_cam_3d(2, 0);

                        /*colored_3d_point.x = -cloud->points[k].y;
                        colored_3d_point.y = -cloud->points[k].z;
                        colored_3d_point.z = cloud->points[k].x;*/

                        colored_3d_point.r = rgb_pixel[2];
                        colored_3d_point.g = rgb_pixel[1];
                        colored_3d_point.b = rgb_pixel[0];
                        out_cloud->points.push_back(colored_3d_point);
                    }
                }
            }
        }

        out_cloud->width = 1;
        out_cloud->height = out_cloud->points.size();
        image_file = rgb_pcl_image_dir_path + buffer.str() + image_file_ext;

        CreateRGB2DImageFromColorfulPCL(out_cloud, image_file);

        /*namedWindow("Display window",
                    WINDOW_AUTOSIZE);     // Create a window for display.
        imshow("Display window", frame);  // Show our image inside it.

        waitKey(0);*/

        std::cout << "Finished Processing of" << i << " images " << endl;
    }
}

int main(int argc, char** argv) {
    ProcessProjection();

    return (0);
}