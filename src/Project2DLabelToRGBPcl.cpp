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

#define RESET "\033[0m"
#define BLACK "\033[30m"  /* Black */
#define RED "\033[31m"    /* Red */
#define GREEN "\033[32m"  /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m"   /* Blue */

std::string log_file_dir = "/home/atas/Msc_Helpers/";

// extensions of files
std::string log_file_ext = ".txt";

void ProcessProjection() {
    std::string full_file = log_file_dir + "log" + log_file_ext;
    std::ifstream calib_file_path_infile(full_file.c_str());

    std::string line;

    while (std::getline(calib_file_path_infile, line)) {
        if (line.find("WARNING", 0) != std::string::npos) {
            std::cout << RED << line << std::endl;
        }
        if (line.find("INFO", 0) != std::string::npos) {
            std::cout << BLUE << line << std::endl;
        }
    }

    if (calib_file_path_infile.is_open()) {
        calib_file_path_infile.close();
    }
}
int main(int argc, char const* argv[]) {
    std::cout << "Hello World" << std::endl;

    ProcessProjection();

    return 0;
}
