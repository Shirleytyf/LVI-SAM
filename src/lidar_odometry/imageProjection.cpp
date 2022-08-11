#include "utility.h"
#include "lvi_sam/cloud_info.h"

// Velodyne
// Velodyne是激光雷达的一种，下面的Ouster也是激光雷达的一种，不过Ouster更适合在无人驾驶的汽车上面使用
struct PointXYZIRT
{
    PCL_ADD_POINT4D // 表示欧几里得xyz坐标 和 强度值的点结构
    PCL_ADD_INTENSITY; // 激光点的反射强度，也可以存点的索引，里面是一个float类型的变量 看到有人说这里的强度可以存放深度信息
    uint16_t ring; // 扫描的激光线
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //这个是eigen的字节对齐的东西,计算机存贮的时候，有的变量字节长，有的短，使用这个就可以使得字节对齐，这样可以提高空间利用率（或者说内存利用率）。
} EIGEN_ALIGN16;

//注册为PCL点云格式
// 一个套路的写法，把上面的结构体里面的变量搬下来

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRT,  
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

// Ouster
// struct PointXYZIRT {
//     PCL_ADD_POINT4D;
//     float intensity;
//     uint32_t t;
//     uint16_t reflectivity;
//     uint8_t ring;
//     uint16_t noise;
//     uint32_t range;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// }EIGEN_ALIGN16;

// POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
//     (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
//     (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
//     (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
// )

const int queueLength = 500; // 队列的长度

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock; // imu队列 、 odom队列互斥锁
    std::mutex odoLock;

    ros::Subscriber subLaserCloud; // 订阅原始激光点云
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud; // 发布当前帧校正后点云，有效点云
    ros::Publisher pubLaserCloudInfo;
    // imu数据队列（原始数据，转lidar系下）
    ros::Subscriber subImu; // 输入imu信息
    std::deque<sensor_msgs::Imu> imuQueue; // deque双端队列
     // imu里程计队列
    ros::Subscriber subOdom; // 输入odometry信息
    std::deque<nav_msgs::Odometry> odomQueue;
    // 激光点云数据队列
    std::deque<sensor_msgs::PointCloud2> cloudQueue; 
    sensor_msgs::PointCloud2 currentCloudMsg; // 队列front帧，作为当前处理帧点云
    // 当前激光帧起止时刻间对应的imu数据，计算相对于起始时刻的旋转增量，以及时间戳;
    // 用于插值计算当前激光帧起止时间范围内，每一时刻的旋转姿态
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse; // Affine3f是仿射变换用到的库

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;  // 当前帧原始激光点云
    pcl::PointCloud<PointType>::Ptr   fullCloud; // 当前帧运动畸变矫正之后的激光点云
    pcl::PointCloud<PointType>::Ptr   extractedCloud; // 从fullcloud中提取的有效点

    int deskewFlag; // deskew 去歪斜
    cv::Mat rangeMat;

    bool odomDeskewFlag; // 当前激光帧起止时刻对应imu里程计位姿变换，该变换对应的平移增量；用于插值计算当前激光帧起止时间范围内，每一时刻的位置
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;
     // 当前帧激光点云运动畸变矫正之后的数据，包括点云数据、初始位姿、姿态角等，发布给featureExtraction进行特征提取
    lvi_sam::cloud_info cloudInfo; // 这里的cloudInfo 是lvi_sam::自己定义的
    double timeScanCur; // 当前帧起始时刻
    double timeScanNext; // 下一帧的开始时刻
    std_msgs::Header cloudHeader; //它包含了时间戳stamp信息，这个在某些函数里面来进行判断用的。


public:
    ImageProjection():
    deskewFlag(0) // 冒号后面是赋值的意思
    {
        //订阅原始imu数据
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        //订阅imu里程计，由imuPreintegration积分计算得到的每时刻imu位姿
        subOdom = nh.subscribe<nav_msgs::Odometry>(PROJECT_NAME + "/vins/odometry/imu_propagate_ros", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        //订阅原始lidar数据
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        //发布当前激光帧运动畸变矫正后的点云，有效点云
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> (PROJECT_NAME + "/lidar/deskew/cloud_deskewed", 5);
        //发布激光点云信息
        pubLaserCloudInfo = nh.advertise<lvi_sam::cloud_info>      (PROJECT_NAME + "/lidar/deskew/cloud_info", 5);
        //初始化，分配内存
        allocateMemory();
        //重置参数
        resetParameters();
        // pcl日志级别，只打ERROR日志
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory() //分配内存
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>()); // 重新reset一个，后面括号里面的和前面的类型对应上就ok
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN); // 这里重新设置大小，就是行列的范围

        cloudInfo.startRingIndex.assign(N_SCAN, 0); // assign 根据函数的意思，这里是给括号里面的变量重新赋值
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0); // 也是重新赋值，这里应该是pointRange的大小，而不是里面的值

        resetParameters(); // 重新设置参数 雷达每帧数据都需要进行重置参数
    }
    //重置参数，接受每帧lidar数据都要重置这些参数
    void resetParameters()
    {
        laserCloudIn->clear(); // 先清空原有的内容
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX)); // 新建一个矩阵

        imuPointerCur = 0; // imu的当前指针？
        firstPointFlag = true; // 第一个点
        odomDeskewFlag = false; // 里程计去歪斜，感觉这个就是去畸变的意思
        // 一个for循环，把下面的内容全部归0，这里的queueLength的长度为500
        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}
    // 简单地接受消息，然后放入imu消息的队列中
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)   //句柄，回调函数？
    {
        // imu原始测量数据转换到lidar系，加速度、角速度、RPY
        // imuConverter（）的实现在utility
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg); // 这里的converter只有旋转没有平移，将imu数据转到lidar系中，就是乘一个矩阵，做一个线性变换
        // 上锁，添加数据的时候队列不可用。执行完函数的时候自动解锁
        std::lock_guard<std::mutex> lock1(imuLock);// 这里加入数据的时候，先上锁，然后再pushback
        // 一个疑问：共享的数据都有哪些  猜测：应该是pushback后面的数据进行上锁，这样在pushback的时候不会被修改
        imuQueue.push_back(thisImu);
    }
    // 订阅imu里程计，由imuPreintegration积分计算得到的每时刻imu位姿
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }
    //回调函数执行的这一套流程，是imageProjection的关键
    // 1.首先是接受数据，计算时间戳，检查数据
    // 2.然后与里程计 和 imu数据进行时间戳同步，并处理
    // 3.接着检查并校准数据
    // 4.提取信息
    // 5.发布校准后的数据，以及提取到的信息
    // 6.重置参数，等待下一次回调函数执行
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {   
        // 添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
        // 将缓存下来多数据进行时间戳的检查、数据的检查、点云密度、扫描线等内容的检查。
        if (!cachePointCloud(laserCloudMsg))
            return;
        // 当前帧起止时刻对应的imu数据、imu里程计数据去畸变处理
        if (!deskewInfo())
            return;
        // 当前帧激光点云运动畸变校正
        // 1.检查激光点距离，扫描线是否合规
        // 2.激光运动畸变校正，保存激光点
        projectPointCloud();
        //提取有效激光点，存extractedCloud
        cloudExtraction();
        //发布当前帧校正后点云，有效点
        publishClouds();
        //重置参数，接受每帧lidar数据都要重置这些参数
        resetParameters();
    }
    // 添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
    // 先判断队列缓存中的数据，再检查点云的密度，看有没有NaN的点
    // 再检查ring channel ，再检查时间戳，这里的时间戳和deskew关系十分密切 最后结束！
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud 缓存点云
        cloudQueue.push_back(*laserCloudMsg);
        //队列缓存中数据过少
        if (cloudQueue.size() <= 2)
            return false;
        else
        {
            // 取出激光点云队列中最早的一帧 取出以后队首被弹出来了
            currentCloudMsg = cloudQueue.front();
            cloudQueue.pop_front();
            //当前帧头部 说明头部存储了时间戳信息
            cloudHeader = currentCloudMsg.header;
            // 当前帧起始时刻
            timeScanCur = cloudHeader.stamp.toSec();
            // 下一帧的开始时刻
            timeScanNext = cloudQueue.front().header.stamp.toSec();
        }

        // convert cloud 转换成pcl点云格式
        pcl::fromROSMsg(currentCloudMsg, *laserCloudIn);

        // check dense flag
        // 存在无效点，Nan 或者Inf
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        // 检查是否存在ring通道，注意static只检查一次
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }     

        // check point time
        // 检查时间戳，以及是否存在time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == timeField)
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
                // 点云时间戳不可用，去偏功能禁用，系统会明显漂移！
        }

        return true;
    }
    // 用于处理激光帧起止时刻对应的imu数据，imu里程计数据
    // 1.用于处理IMU数据
    // 2.用于处理IMU里程计数据
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock); // 处理数据之前先上锁，避免影响处理的数据
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        // 确保IMU数据的时间戳包含了整个lidar数据的时间戳，否则就不处理了
        // imu不能为空，并且imu队首的时间要比timeScanCur要提前，队尾的时间要比timeScanNext要晚，即imu的时间跨度更大一些
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanNext)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }
        // 下面就是分别对imu 和 odom进行去歪斜
        // 当前帧对应imu数据处理
        // 注：imu数据都已经转换到lidar系下了
        imuDeskewInfo();
        // 当前帧对应imu里程计处理
        // 注：imu数据都已经转换到lidar系下了
        odomDeskewInfo();

        return true;
    }
    // 遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
    // 然后用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0.
    // 物体姿态计算，是由这个函数来完成的
    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false; // 先把available变量设置为false
        // 删除imu队列中，当前lidar数据的时间戳的0.01s前的数据
        // 进行一个while循环，把早于timeScanCur0.01的数据进行pop掉。
        // 这里用了pop_front 而不是pop 这就是它的双端队列进行pop的操作，还是有区别的
        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }
        // 如果imu缓存队列中没有数据，直接返回
        if (imuQueue.empty())
            return;
        // 处理的imu的帧数，或者说游标
        // 这个就是记录下面数据的
        // 这个变量想当个指针，来存放当前imu帧处理到第几个了
        imuPointerCur = 0;
        // 遍历当前lidar数据帧起止时刻（前后拓展0.01s）之间的imu数据
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i]; // 提取imu信息
            double currentImuTime = thisImuMsg.header.stamp.toSec(); // 提取时间，toSec这个函数很人性化，转换成s，统一计算

            // get roll, pitch, and yaw estimation for this scan 提取imu姿态角RPY，作为当前lidar帧初始姿态角
            if (currentImuTime <= timeScanCur)
                // 这个imuRPY to rosRPY就是一个数据的转换
                // 判断条件：当前imu的时间戳，小于雷达的时间。scan表示了雷达的扫描
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);
            // 超过当前lidar数据的时间戳结束时刻0.01s，结束
            if (currentImuTime > timeScanNext + 0.01)
                break;
            // 第一帧imu旋转角初始化
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur; // 指针加1
                continue; // 结束本次大的循环，然后++i
            }

            // get angular velocity 提取imu角速度
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation 对旋转进行积分
            // imu帧间时差
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1]; // 计算一个时间差
            // 当前时刻旋转角 = 前一时刻旋转角 + 角速度 * 时差
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }
        // 去除处理的第一帧的初始化，用于下面的判断
        --imuPointerCur;
        // 没有合规的imu数据
        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true; // 最后让available的值为true
    }
    // 遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
    // 并用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
    //求odom的信息，里面求了最初的，然后求了现在的，根据矩阵乘积，得到一个转换矩阵，这里的话在xyz的基础上加了RPY。需要注意的是通过四元数来求得RPY，可以理解是做了一个预测，这里计算的值会在下面的部分用到。
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;
        // 删除odom队列中，当前时间戳的0.01s前的数据
        // 和imu一样的判断
        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }
        // 如果删除后为空，直接返回
        if (odomQueue.empty())
            return;
        // 必须包含当前lidar数据的时间戳之前的odom数据，即imu的时间跨度 必须比 雷达的要大
        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan提取当前lidar数据的时间戳
        // 下面这个for循环是找到timeScanCur最近的那个odom的信息
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }
        // 提取imu里程计姿态角
        tf::Quaternion orientation; // 四元数
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation); // 朝向 RPY 一个四元数的转换，这个startOdomMsg是上面for循环得到的

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw); // 获取RPY

        // Initial guess used in mapOptimization
        // 用当前激光帧起始时刻的odom，初始化lidar位姿，后面用于mapOptimization
        cloudInfo.odomX = startOdomMsg.pose.pose.position.x;
        cloudInfo.odomY = startOdomMsg.pose.pose.position.y;
        cloudInfo.odomZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.odomRoll  = roll;
        cloudInfo.odomPitch = pitch;
        cloudInfo.odomYaw   = yaw;
        cloudInfo.odomResetId = (int)round(startOdomMsg.pose.covariance[0]);

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan 获取最后的odom
        odomDeskewFlag = false;
        // 如果当前激光帧结束时刻之后没有odom数据，直接返回
        if (odomQueue.back().header.stamp.toSec() < timeScanNext)
            return;
        // 提取当前激光帧结束时刻的odom
        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanNext)
                continue;
            else
                break;
        }
        // 如果起止时刻对应odom的协方差不等，可能是里程计出现了问题，它们两个的相关性应该是一致的，所以直接返回
        // 这里存疑：covariance[0]表示的是啥
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;
        // 坐标系之间的转换--仿射变换
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw); // 先转换成四元数，然后得到RPY
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        // 起止时刻imu里程计的相对变换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;
        // 相对变换，提取增量平移、旋转（欧拉角）
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }
    // 在当前激光帧起止时间范围内，计算某一时刻的旋转（相对于起始时刻的旋转增量）
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;
        // 查找当前时刻在imuTime下的索引
        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }
        // 设为离当前时刻最近的旋转增量
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            // 前后时刻插值计算当前时刻的旋转增量
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }
    // 在当前激光帧起止时间范围内，计算某一时刻的平移（相对于起始时刻的平移增量）
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        // 如果传感器移动速度较慢，例如人行走的速度，那么可以认为激光在一帧时间范围内，平移量可以小到忽略不计
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanNext - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }
     // 用当前帧起止时刻之间的imu数据计算旋转增量，imu里程计数据计算平移增量
    // 进而将每一时刻激光点位置变换到第一个激光点坐标系下，进行运动补偿。
    // 在上面的代码片段中，实现了激光运动畸变校正的功能，通俗来讲，就是将在运动过程中的获取到基于运动时的位置的点图，转变成在最初始的位置的点图，其中需要用到imu信息进行转变。而用来计算这些转变时条用了Eigen和pcl库进行计算。不过作者也定义了两个简单的函数去计算旋转增量 和 平移增量 findRotation() 和 findPositon()
    PointType deskewPoint(PointType *point, double relTime)
    {
        // 检查是否有时间戳信息，和是否有合规的imu数据
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;
        // relTime是当前激光点相对于激光帧起始时刻的时间，pointTime则是当前激光点的时间戳
        double pointTime = timeScanCur + relTime;
        // 在当前激光帧起止时间范围内，计算某一时刻的旋转（相对于起始时刻的旋转增量）
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);
        // 在当前激光帧起止时间范围内，计算某一时刻的平移（相对于起始时刻的平移增量）
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);
        // 第一个点的位姿增量（theta），求逆
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        // 当前时刻激光点与第一个激光点的位姿变换
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;
        // 当前激光点在第一个激光点坐标系下的坐标
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
        
    }
    // 检查雷达数据并校正
    void projectPointCloud()
    {
        int cloudSize = (int)laserCloudIn->points.size(); // 获取激光点云的大小
        // range image projection 遍历当前帧激光点云
        for (int i = 0; i < cloudSize; ++i)
        {
            // pcl格式
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            // 扫描线检查
            // 这个ring 存放的应该是扫描的条数
            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            // 扫描线如果有降采样，则跳过
            // 看了一下 downsampleRate取值为1
            if (rowIdn % downsampleRate != 0)
                continue;

            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI; // 获取水平的角度范围
            // 水平扫描角度步长，例如一周扫描1800次，则两次扫描间隔角度0.2°
            static float ang_res_x = 360.0/float(Horizon_SCAN);
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
            // 计算到lidar的距离，具体实现在utility.h
            float range = pointDistance(thisPoint);
            // 所以这个range可以表示深度，也可以表示距离
            // 如果距离小于一个阈值，则跳过该点，比如说扫到手持设备的人
            if (range < 1.0)
                continue;
            // 已经存过该点，不再处理
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            // for the amsterdam dataset
            // 这个针对阿姆斯特丹数据集，有特别的判断条件，这个数据集没看到
            // if (range < 6.0 && rowIdn <= 7 && (columnIdn >= 1600 || columnIdn <= 200))
            //     continue;
            // if (thisPoint.z < -2.0)
            //     continue;
            // 矩阵存激光点的距离
            rangeMat.at<float>(rowIdn, columnIdn) = range; //extract的条件，如果有值就push，没有值为FLT_MAX的时候就不会push进来。
            // 激光运动畸变校正
            // 利用当前帧起止时刻之间的imu数据计算旋转增量，imu里程计数据计算平移增量，
            // 进而将每一时刻激光点位置变换到第一个激光点坐标系下，进行运动补偿
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time); // Velodyne
            // thisPoint = deskewPoint(&thisPoint, (float)laserCloudIn->points[i].t / 1000000000.0); // Ouster
            // 这个地方出现了Velodyne 和 Ouster 说明lvi-sam还可以用来做Ouster雷达的 这个通过资料可知，这个雷达最常用在无人驾驶上面

            // 转换成一维索引，存校正之后的激光点
            // 这里可以看出index 是按照行号进行排的
            int index = columnIdn  + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }
    // 提取有效激光点，存extractedCloud
    // 这里的有效点，初步理解 rangeMat.at<float>(i,j)被上面的函数中赋值的点称为有效点
    // 并且这里记录了雷达每根扫描线的第5个点 和 倒数第5个点 （有种算曲率的意思）
    void cloudExtraction()
    {
        // 有效激光点数量
        int count = 0;
        // extract segmented cloud for lidar odometry
        // 为激光雷达测距提取分割的云层，遍历所有点
        for (int i = 0; i < N_SCAN; ++i)
        {
            // 记录每根扫描线起始第5个激光点在一维数组中的索引
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                // 有效激光点
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    // 标记点的列索引，以便以后标记能对应上
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    // 保存范围信息
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    // 保存提取出来的点云
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]); //是点云的一维存储方式，按行来存储，就是一行一行来的。前面那篇文章的问题中的其中一个，想着应该是解决了。
                    // size of extracted cloud
                    // 增加点云的范围
                    ++count;
                }
            }
            // 记录每根扫描线倒数第5个激光点在一维数组中的索引
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader; // header里面有时间戳
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, "base_link");
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Lidar Cloud Deskew Started.\033[0m"); // 激光雷达云校正开始 

    ros::MultiThreadedSpinner spinner(3); // 多线程
    spinner.spin(); // 这个地方就是来调用回调函数的
    
    return 0;
}