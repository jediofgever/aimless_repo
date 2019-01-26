/**
 * @author Fetullah Atas
 * @email [fetullah.atas@mindtronicai.com]
 * @create date 2018-10-16 15:06:28
 * @modify date 2018-10-16 15:06:28
 * @desc [description]
 */
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>

int user_data;

void viewerOneOff(pcl::visualization::PCLVisualizer& viewer) {
    viewer.setBackgroundColor(1.0, 0.5, 1.0);
    pcl::PointXYZ o;
    o.x = 1.0;
    o.y = 0;
    o.z = 0;
    viewer.addSphere(o, 0.25, "sphere", 0);
    std::cout << "i only run once" << std::endl;
}

void viewerPsycho(pcl::visualization::PCLVisualizer& viewer) {
    static unsigned count = 0;
    std::stringstream ss;
    ss << "Once per viewer loop: " << count++;
    viewer.removeShape("text", 0);
    viewer.addText(ss.str(), 200, 300, "text", 0);

    // FIXME: possible race condition here:
    user_data++;
}

int main() {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile("test_pcd.pcd", *cloud);
    std::cout << cloud->points.size() << std::endl;

    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");

    viewer.showCloud(cloud);

    while (!viewer.wasStopped()) {
    }

    return 0;
}
