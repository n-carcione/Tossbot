import numpy as np
import math

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import min_jerk

import rospy
import cv2
from cv_bridge import CvBridge
from perception import CameraIntrinsics
from utils import *
from toss_bot_perception import *
import tossbot_utils as tu


if __name__ == "__main__":
    fa = FrankaArm()
    while True:
        fa.stop_skill()
        fa.reset_joints()
        fa.close_gripper()

        #GET STUFF FROM PERCEPTION
        '''============================================================================'''
        print('Opening the eyes!!')

        rospy.sleep(1)
        AZURE_KINECT_INTRINSICS = 'calib_tossbot/azure_kinect.intr'
        AZURE_KINECT_EXTRINSICS = 'calib_tossbot/azure_kinect_overhead/azure_kinect_overhead_to_world.tf'

        cv_bridge = CvBridge()
        azure_kinect_intrinsics = CameraIntrinsics.load(AZURE_KINECT_INTRINSICS)
        azure_kinect_to_world_transform = RigidTransform.load(AZURE_KINECT_EXTRINSICS)

        (center,target_world_frame,azure_kinect_rgb_image,contour) = locate_container(cv_bridge,
                                                                        azure_kinect_intrinsics,
                                                                        azure_kinect_to_world_transform)
                
        print(f"Transformed\nx: {target_world_frame[0]}, y: {target_world_frame[1]}")

        # draw center on image
        azure_kinect_rgb_image = cv2.circle(azure_kinect_rgb_image, (int(center[0]),int(center[1])), radius=10, color=(255, 255, 255), thickness=-1)

        # draw the biggest contour (c) in green
        #cv2.rectangle(red_mask,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawContours(azure_kinect_rgb_image, contour, -1, (0,0,0), 20)


        cv2.namedWindow("image")
        azure_kinect_rgb_image = cv2.resize(azure_kinect_rgb_image, (1160, 740))  
        cv2.imshow("image", azure_kinect_rgb_image)
        #cv2.setMouseCallback("image", onMouse, object_image_position)
        cv2.waitKey()

        '''========================================================================'''
        target_location = target_world_frame
        theta_1 = tu.get_joint_1(target_location)

        #Joint angles 1, 3, 5, and 7 are the same as the reset angles
        #Only change joint angles 2, 4, 6
        stationary_joints = [1,3,5,7]
        moving_joints = [2,4,6]
        #Underhand
        joints_init = [theta_1, -0.4431077, 0.0, -2.86279262, 0.0, 1.80853718, 7.85244335e-01]
        joints_rel = [theta_1, 0.11036068, 0.0, -2.07947407, 0.0, 2.65808482, 7.85244335e-01]
        transforms = fa.get_links_transforms(joints_rel, use_rigid_transforms=False)
        ee_pos = transforms[-1, :3, 3]
        rj = fa.get_jacobian(joints_rel)

        dist = math.sqrt( (target_location[0]-ee_pos[0])**2 + (target_location[1]-ee_pos[1])**2 )

        vel_rel = tu.get_release_velocity(dist, theta_1, ee_pos[2])
        print(vel_rel)
        release_joint_vel = tu.get_release_joint_velocity(vel_rel, rj, stationary_joints, moving_joints)
        print(release_joint_vel)
        tr = 1.0
        tf = 2.0
        dt = 0.01

        q20 = joints_init[1]
        q2r = joints_rel[1]
        q2f = 0.36
        q2r_dot = release_joint_vel[1]

        q40 = joints_init[3]
        q4r = joints_rel[3]
        q4f = -1.6
        q4r_dot = release_joint_vel[3]

        q60 = joints_init[5]
        q6r = joints_rel[5]
        q6f = 3.1
        q6r_dot = release_joint_vel[5]

        Ar = [[0, 0, 0, 1], [0, 0, 1, 0], [tr**3, tr**2, tr, 1], [3*tr**2, 2*tr, 1, 0]]
        Af = [[tr**3, tr**2, tr, 1], [3*tr**2, 2*tr, 1, 0], [tf**3, tf**2, tf, 1], [3*tf**2, 2*tf, 1, 0]]

        b2r = [q20, 0, q2r, q2r_dot]
        b2f = [q2r, q2r_dot, q2f, 0]
        b4r = [q40, 0, q4r, q4r_dot]
        b4f = [q4r, q4r_dot, q4f, 0]
        b6r = [q60, 0, q6r, q6r_dot]
        b6f = [q6r, q6r_dot, q6f, 0]

        c2r = np.linalg.inv(Ar) @ b2r
        c2f = np.linalg.inv(Af) @ b2f
        c4r = np.linalg.inv(Ar) @ b4r
        c4f = np.linalg.inv(Af) @ b4f
        c6r = np.linalg.inv(Ar) @ b6r
        c6f = np.linalg.inv(Af) @ b6f

        joints_traj = []

        q1 = joints_init[0]
        q3 = joints_init[2]
        q5 = joints_init[4]
        q7 = joints_init[6]

        for t in np.arange(0,tf,dt):
            if (t <= tr):
                q2= c2r[0]*t**3 + c2r[1]*t**2 + c2r[2]*t + c2r[3] 
                q4 = c4r[0]*t**3 + c4r[1]*t**2 + c4r[2]*t + c4r[3]
                q6 = c6r[0]*t**3 + c6r[1]*t**2 + c6r[2]*t + c6r[3]
                joints_traj.append([q1, q2, q3, q4, q5, q6, q7])
            else:
                q2 = c2f[0]*t**3 + c2f[1]*t**2 + c2f[2]*t + c2f[3]
                q4 = c4f[0]*t**3 + c4f[1]*t**2 + c4f[2]*t + c4f[3]
                q6 = c6f[0]*t**3 + c6f[1]*t**2 + c6f[2]*t + c6f[3]
                joints_traj.append([q1, q2, q3, q4, q5, q6, q7])
        
        T = 2
        dt = 0.01
        ts = np.arange(0, T, dt)

        fa.goto_joints(joints_traj[0])
        rospy.loginfo('Initializing Sensor Publisher')
        pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        rate = rospy.Rate(1 / dt)

        rospy.loginfo('Publishing joints trajectory...')
        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        fa.goto_joints(joints_traj[0], duration=T, dynamic=True, buffer_time=10)
        init_time = rospy.Time.now().to_time()
        for i in range(1, len(ts)-1):
            traj_gen_proto_msg = JointPositionSensorMessage(
                id=i, timestamp=rospy.Time.now().to_time() - init_time, 
                joints=joints_traj[i]
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
            )

            if i == 100:
                fa.goto_gripper(0.08, speed=1.2 , block=False)
            
            rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
            pub.publish(ros_msg)
            rate.sleep()

        # Stop the skill
        # Alternatively can call fa.stop_skill()
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )

        pub.publish(ros_msg)

        rospy.loginfo('Done')

    