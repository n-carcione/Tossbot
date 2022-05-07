import math
import numpy as np

def get_joint_1(target_location):
    # Use the x- and y-values of the target in the robot-base frame to calculate
    # what the value for the first joint should be
    return math.atan2(target_location[1]-0.02,target_location[0])

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

def get_ball_grab_pose(ball_loc):
    reset_rot_mat = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    ball_pose = np.zeros([4,4])
    ball_pose[:3,:3] = reset_rot_mat
    #x-position should have a 0.01 offset as determined by testing
    ball_pose[0, 3] = ball_loc[0] + 0.01
    #y-position needs a 0.02 offset as determined by testing
    ball_pose[1, 3] = ball_loc[1] + 0.02
    #Ball will always be at this height as defined by set up
    ball_pose[2, 3] = 0.025
    ball_pose[3,3] = 1

    return ball_pose.flatten()

def check_joint_vels(vels):
    vel_constraints = [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
    for i in range(len(vels)):
        if vels[i] > vel_constraints[i]:
            print('Velocity constraint at joint {}'.format(i+1))
            print('Joint {} requires velocity of {}'.format(i+1, vels[i]))
            return False
    return True

def get_endpoints(joints_init, joints_rel, release_joint_vel):
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

    endpoints = [q20, q2r, q2f, q2r_dot, q40, q4r, q4f, q4r_dot, q60, q6r, q6f, q6r_dot]

    return endpoints

def get_A(tr, tf):
    Ar = [[0, 0, 0, 1], [0, 0, 1, 0], [tr**3, tr**2, tr, 1], [3*tr**2, 2*tr, 1, 0]]
    Af = [[tr**3, tr**2, tr, 1], [3*tr**2, 2*tr, 1, 0], [tf**3, tf**2, tf, 1], [3*tf**2, 2*tf, 1, 0]]

    A = [Ar, Af]
    return A

def get_b(ep):
    b2r = [ep[0], 0, ep[1], ep[3]]
    b2f = [ep[1], ep[3], ep[2], 0]
    b4r = [ep[4], 0, ep[5], ep[7]]
    b4f = [ep[5], ep[7], ep[6], 0]
    b6r = [ep[8], 0, ep[9], ep[11]]
    b6f = [ep[9], ep[11], ep[10], 0]

    b = [b2r, b2f, b4r, b4f, b6r, b6f]

    return b

def calc_c(A, b):
    c2r = np.linalg.inv(A[0]) @ b[0]
    c2f = np.linalg.inv(A[1]) @ b[1]
    c4r = np.linalg.inv(A[0]) @ b[2]
    c4f = np.linalg.inv(A[1]) @ b[3]
    c6r = np.linalg.inv(A[0]) @ b[4]
    c6f = np.linalg.inv(A[1]) @ b[5]

    c = [c2r, c2f, c4r, c4f, c6r, c6f]

    return c

def gen_polynomials(joints_init, joints_rel, release_joint_vel, tr):
    tf = tr + 1.0
    dt = 0.01

    endpoints = get_endpoints(joints_init, joints_rel, release_joint_vel)

    A = get_A(tr, tf)
    b = get_b(endpoints)
    c = calc_c(A,b)

    joints_traj = []

    q1 = joints_init[0]
    q3 = joints_init[2]
    q5 = joints_init[4]
    q7 = joints_init[6]

    for t in np.arange(0,tf,dt):
        if (t <= tr):
            q2 = c[0][0]*t**3 + c[0][1]*t**2 + c[0][2]*t + c[0][3] 
            q4 = c[2][0]*t**3 + c[2][1]*t**2 + c[2][2]*t + c[2][3]
            q6 = c[4][0]*t**3 + c[4][1]*t**2 + c[4][2]*t + c[4][3]
            joints_traj.append([q1, q2, q3, q4, q5, q6, q7])
        else:
            q2 = c[1][0]*t**3 + c[1][1]*t**2 + c[1][2]*t + c[1][3]
            q4 = c[3][0]*t**3 + c[3][1]*t**2 + c[3][2]*t + c[3][3]
            q6 = c[5][0]*t**3 + c[5][1]*t**2 + c[5][2]*t + c[5][3]
            joints_traj.append([q1, q2, q3, q4, q5, q6, q7])

    return (joints_traj, c, tf)

def exists_vel_violation(c, tr):
    tf = tr + 1.0
    dt = 0.01

    for t in np.arange(0,tf,dt):
        if (t <= tr):
            q2dot = 3*c[0][0]*t**2 + 2*c[0][1]*t + c[0][2]
            q4dot = 3*c[2][0]*t**2 + 2*c[2][1]*t + c[2][2]
            q6dot = 3*c[4][0]*t**2 + 2*c[4][1]*t + c[4][2]
            if q2dot > 2.175 or q4dot > 2.175 or q6dot > 2.61:
                return True
        else:
            q2dot = 3*c[1][0]*t**2 + 2*c[1][1]*t + c[1][2]
            q4dot = 3*c[3][0]*t**2 + 2*c[3][1]*t + c[3][2]
            q6dot = 3*c[5][0]*t**2 + 2*c[5][1]*t + c[5][2]
            if q2dot > 2.175 or q4dot > 2.175 or q6dot > 2.61:
                return True

    return False