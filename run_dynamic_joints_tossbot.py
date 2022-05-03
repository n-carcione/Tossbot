import numpy as np
import math

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import convert_array_to_rigid_transform

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
    thrown = True
    AZURE_KINECT_INTRINSICS = 'calib_tossbot/azure_kinect.intr'
    AZURE_KINECT_EXTRINSICS = 'calib_tossbot/azure_kinect_overhead/azure_kinect_overhead_to_world.tf'

    cv_bridge = CvBridge()
    azure_kinect_intrinsics = CameraIntrinsics.load(AZURE_KINECT_INTRINSICS)
    azure_kinect_to_world_transform = RigidTransform.load(AZURE_KINECT_EXTRINSICS)
    while True:
        fa.stop_skill()
        fa.reset_joints()
        fa.open_gripper()

        #GET STUFF FROM PERCEPTION
        '''============================================================================'''
        rospy.sleep(1)

        (ball_centers, ball_world_frame, azure_kinect_rgb_image,ball_contours) = locate_balls(cv_bridge, azure_kinect_intrinsics, azure_kinect_to_world_transform)

        #Only execute ball pickup and throw if there is a ball detected
        while ball_centers:
            fa.stop_skill()
            fa.reset_joints()
            if thrown:
                (ball_centers, ball_world_frame, azure_kinect_rgb_image, ball_contours) = locate_balls(cv_bridge, azure_kinect_intrinsics, azure_kinect_to_world_transform)
                #Show all the ball detections
                for bc in ball_centers:
                    all_balls_rgb_image = cv2.circle(azure_kinect_rgb_image, (int(bc[0]),int(bc[1])), radius=10, color=(255, 255, 255), thickness=-1)
                for cont in ball_contours:
                    cv2.drawContours(all_balls_rgb_image, cont, -1, (0,0,0), 20)
                all_balls_rgb_image = cv2.resize(all_balls_rgb_image, (600, 400))  
                cv2.imshow("image", all_balls_rgb_image)
                cv2.waitKey(1500)
            #Only consider something to be a ball if it is 0.1m - 0.8m in front of the robot
            #This helps avoid false detections from the slightly green 8020 used to 
            #construct the table for the arm
            if ball_world_frame[0][0] > 0.1 and ball_world_frame[0][0] < 0.5 and ball_world_frame[0][1] > -0.4 and ball_world_frame[0][1] < 0.4:                
                #draw targetted ball on image
                target_ball_rgb_image = cv2.circle(azure_kinect_rgb_image, (int(ball_centers[0][0]),int(ball_centers[0][1])), radius=10, color=(255, 255, 255), thickness=-1)
                cv2.drawContours(target_ball_rgb_image, ball_contours[0], -1, (0,255,255), 20)
                cv2.namedWindow("image")
                target_ball_rgb_image = cv2.resize(target_ball_rgb_image, (600, 400))  
                cv2.imshow("image", target_ball_rgb_image)
                cv2.waitKey(1500)
                
                (center,target_world_frame,azure_kinect_rgb_image,contour) = locate_container(cv_bridge,
                                                                                azure_kinect_intrinsics,
                                                                                azure_kinect_to_world_transform)

                # draw detected bin on image
                azure_kinect_rgb_image = cv2.circle(azure_kinect_rgb_image, (int(center[0]),int(center[1])), radius=10, color=(255, 255, 255), thickness=-1)
                cv2.drawContours(azure_kinect_rgb_image, contour, -1, (0,0,0), 20)
                cv2.namedWindow("image")
                azure_kinect_rgb_image = cv2.resize(azure_kinect_rgb_image, (600, 400))  
                cv2.imshow("image", azure_kinect_rgb_image)
                cv2.waitKey(1500)
                '''========================================================================'''

                target_location = target_world_frame
                theta_1 = tu.get_joint_1(target_location)

                #Joint angles 1, 3, 5, and 7 are the same as the reset angles
                #Only change joint angles 2, 4, 6
                stationary_joints = [1,3,5,7]
                moving_joints = [2,4,6]
                #Starting and release joint values for an underhand throw
                joints_init = [theta_1, -0.4431077, 0.0, -2.86279262, 0.0, 1.80853718, 7.85244335e-01]
                joints_rel = [theta_1, 0.11036068, 0.0, -2.07947407, 0.0, 2.65808482, 7.85244335e-01]
                #Compute Jacobian at release
                rj = fa.get_jacobian(joints_rel)

                #Compute forward kinematics at release position to find (x,y) position
                #of end effector.  Use this to calculate distance ball needs to be thrown
                transforms = fa.get_links_transforms(joints_rel, use_rigid_transforms=False)
                ee_pos = transforms[-1, :3, 3]
                dist = math.sqrt( (target_location[0]-ee_pos[0])**2 + (target_location[1]-ee_pos[1])**2 )
                print('Distance to throw: {}m'.format(dist))
                
                vel_rel = tu.get_release_velocity(dist, theta_1, ee_pos[2])
                release_joint_vel = tu.get_release_joint_velocity(vel_rel, rj, stationary_joints, moving_joints)
                
                # Check to make sure the target is within the throwing range (joint velocities don't
                # violate defined constraints)
                if not tu.check_joint_vels(release_joint_vel):
                    print("Target out of range")
                    thrown = False
                    fa.reset_joints()
                    break

                #Grab the first ball detected
                ball_pose = tu.get_ball_grab_pose(ball_world_frame[0])
                fa.goto_pose(convert_array_to_rigid_transform(ball_pose),ignore_virtual_walls=True)
                fa.close_gripper()

                tr = 1.0
                (joints_traj, c, T) = tu.gen_polynomials(joints_init, joints_rel, release_joint_vel, tr)
                
                while tu.exists_vel_violation(c, tr):
                    tr = tr + 0.1
                    (joints_traj, c, T) = tu.gen_polynomials(joints_init, joints_rel, release_joint_vel, tr)


                thrown = True
                ball_centers.remove(ball_centers[0])
                ball_world_frame.remove(ball_world_frame[0])
                ball_contours.remove(ball_contours[0])

                dt = 0.01
                ts = np.arange(0, T, dt)

                fa.goto_joints(joints_traj[0])
                # rospy.loginfo('Initializing Sensor Publisher')
                pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
                rate = rospy.Rate(1 / dt)

                # rospy.loginfo('Publishing joints trajectory...')
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

                    #Release point is actually at i==100 but there is a slight delay
                    #Reasonable performance found when i==85
                    if i == 85:
                        fa.goto_gripper(0.08, speed=1.2 , block=False)
                    
                    pub.publish(ros_msg)
                    rate.sleep()
            else:
                thrown = False
                ball_centers.remove(ball_centers[0])
                ball_world_frame.remove(ball_world_frame[0])
                ball_contours.remove(ball_contours[0])