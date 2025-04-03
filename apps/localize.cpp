#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pclomp/ndt_omp.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl/common/transforms.h>
#include <tf/transform_broadcaster.h>

pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt_omp;
pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZ>());
tf::TransformListener* tf_listener;
tf::TransformBroadcaster* tf_broadcaster;
ros::Publisher map_pub;
ros::Publisher aligned_pub;

static Eigen::Matrix4f last_transform = Eigen::Matrix4f::Identity();

void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *source);

    tf::StampedTransform tf_transform;
    try {
        tf_listener->waitForTransform("base", msg->header.frame_id, ros::Time(0), ros::Duration(0.5));
        tf_listener->lookupTransform("base", msg->header.frame_id, ros::Time(0), tf_transform);
    } catch (tf::TransformException& ex) {
        ROS_WARN_STREAM("TF failed: " << ex.what());
        return;
    }

    // pointcloud to base tf
    Eigen::Affine3d eigen_transform;
    tf::transformTFToEigen(tf_transform, eigen_transform);
    Eigen::Matrix4f tf_mat = eigen_transform.cast<float>().matrix();
    pcl::transformPointCloud(*source, *source, tf_mat);

    // Downsample source cloud to match map resolution
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setLeafSize(0.2f, 0.2f, 0.2f);
    voxel.setInputCloud(source);
    voxel.filter(*source);

    ndt_omp->setInputSource(source);
    pcl::PointCloud<pcl::PointXYZ> aligned;
    ndt_omp->align(aligned, last_transform);

    if (!ndt_omp->hasConverged()) {
        ROS_WARN("NDT did not converge");
        return;
    }

    last_transform = ndt_omp->getFinalTransformation();

    double fitness = ndt_omp->getFitnessScore();
    ROS_INFO_STREAM("Fitness score: " << fitness);

    Eigen::Vector3f t = last_transform.block<3,1>(0,3);
    Eigen::Matrix3f R = last_transform.block<3,3>(0,0);
    Eigen::Quaternionf q(R);

    // ROS_INFO_STREAM("Transform (map -> base):\nTranslation: " << t.transpose() << "\nRotation: " << q.coeffs().transpose());

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(t.x(), t.y(), t.z()));
    transform.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));

    tf_broadcaster->sendTransform(tf::StampedTransform(
        transform,
        msg->header.stamp,
        "map",
        "base"
    ));

    sensor_msgs::PointCloud2 aligned_msg;
    pcl::toROSMsg(aligned, aligned_msg);
    aligned_msg.header.frame_id = "map";
    aligned_msg.header.stamp = msg->header.stamp;
    aligned_pub.publish(aligned_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "ndt_localizer_node");
    ros::NodeHandle nh("~");

    tf_broadcaster = new tf::TransformBroadcaster();

    std::string map_path;
    nh.param<std::string>("map_path", map_path, "/home/emir/slam_ws/src/ndt_omp/data/map.pcd");
    double resolution;
    nh.param("voxel_leaf_size", resolution, 0.3);

    if (pcl::io::loadPCDFile(map_path, *map_cloud) < 0) {
        ROS_ERROR_STREAM("Failed to load map file: " << map_path);
        return -1;
    }

    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(map_cloud);
    voxel.setLeafSize(resolution, resolution, resolution);
    voxel.filter(*map_cloud);

    ndt_omp.reset(new pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
    ndt_omp->setResolution(1.0);
    ndt_omp->setNumThreads(omp_get_max_threads());
    ndt_omp->setNeighborhoodSearchMethod(pclomp::KDTREE);
    ndt_omp->setInputTarget(map_cloud);

    tf_listener = new tf::TransformListener();

    map_pub = nh.advertise<sensor_msgs::PointCloud2>("/map_cloud", 1, true);
    aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_cloud", 1);

    sensor_msgs::PointCloud2 map_msg;
    pcl::toROSMsg(*map_cloud, map_msg);
    map_msg.header.frame_id = "map";
    map_msg.header.stamp = ros::Time::now();
    map_pub.publish(map_msg);

    ros::Subscriber sub = nh.subscribe("/cloud_all_fields_fullframe", 1, cloudCallback);
    ros::spin();

    delete tf_listener;
    delete tf_broadcaster;
    return 0;
}
