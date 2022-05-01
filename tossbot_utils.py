import math
import numpy as np

def get_joint_1(target_location):
    # Use the x- and y-values of the target in the robot-base frame to calculate
    # what the value for the first joint should be
    return math.atan2(target_location[1],target_location[0])

def get_release_velocity(d, theta_1, h0):
    # Calculate the necessary release velocity so that the ball will
    #travel a total distance of d to the target
    # Defined parameters (arbitrarily chosen) and constants
    g = 9.81         #m/s^2
    a = math.pi / 4  #rad

    v0 = (d/math.cos(a)) * math.sqrt( (0.5*g) / (h0 + d*math.tan(a)) )
    vel_release = [v0*math.cos(a)*math.cos(theta_1), v0*math.cos(a)*math.sin(theta_1), v0*math.sin(a), 0.0, -3.0, 0.0]
    return vel_release

def get_release_joint_velocity(vel_rel, rj, stationary_joints, moving_joints):
    rjt = np.transpose(rj)
    rji = rjt @ np.linalg.inv(rj @ rjt)

    release_joint_vel = rji @ vel_rel

    for stationary_joint in stationary_joints:
        release_joint_vel[stationary_joint-1] = 0
    
    # for moving_joint in moving_joints:
#     if abs(release_joint_vel[moving_joint-1]) > max_speeds_moving_joints[moving_joint-1]:
#         if release_joint_vel[moving_joint-1] > 0:
#             release_joint_vel[moving_joint-1] = max_speeds_moving_joints[moving_joint-1]
#         else:
#             release_joint_vel[moving_joint-1] = -1*max_speeds_moving_joints[moving_joint-1]


    return release_joint_vel