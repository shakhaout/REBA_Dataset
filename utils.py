import numpy as np
from random import randint
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys, os


def show_skeleton(joints, title=None):
    '''
    Input: 17 keypoints, title of graph
    plot human pose in 3D plot
    '''
    if joints.shape[1] == 17:  # Human 3.6m dataset
        bone_list = [[0,1], [0,4], [0,7], [1,2], [2,3], [4,5], [5,6], [7,8], [8,9], [8,11], [8,14], 
                      [9,10], [11,12], [12,13], [14,15], [15,16]]

    else:
        print('Input the correct data with shape 17x3')
        
    plt.figure(figsize=(7,7))
    ax = plt.axes(projection='3d')
    if title is not None:
        plt.title(title, y=0.95)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    pose3D = joints.transpose()
    X = pose3D[0, :]
    Y = pose3D[2, :]
    Z = -pose3D[1, :]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y + max_range, mid_y - max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
#     ax.axes.xaxis.set_ticklabels([])
#     ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.view_init(azim =-90, elev=-100)

    for joint in joints:
#         color = (randint(64, 255) / 255, randint(64, 255) / 255, randint(64, 255) / 255)
        x, y, z = joint.T[0], joint.T[1], joint.T[2]

        ax.scatter3D(x, y, z, color=(0,0,1))
        for bone in bone_list:
            ax.plot3D([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], [z[bone[0]], z[bone[1]]], color=(1,0,1))
    
#     plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[], zticks=[])
    plt.grid(True)
    plt.show()

    
def draw_pose(argv,Title,fign):
    '''
    Draw pose
    '''
    try:
        pose3D = argv
        buff_large = np.zeros((32, 3))
        buff_large[(0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27), :] = pose3D

        pose3D = buff_large.transpose()
        
        kin = np.array([[0, 12], [12, 13], [13, 14], [15, 14], [13, 17], [17, 18], [18, 19],
                        [13, 25], [25, 26], [26, 27], [0, 1], [1, 2], [2, 3], [0, 6], [6, 7], [7, 8]])
        order = np.array([0,2,1])

        mpl.rcParams['legend.fontsize'] = 20

        fig = plt.figure(figsize=(15,15))
        ax = plt.axes(projection='3d')
        ax.view_init(azim=-90, elev=25)
        
        for link in kin:
            ax.plot(pose3D[0, link], pose3D[2, link], -pose3D[1, link], linewidth=2.0,color=(1,0,1))
#             ax.scatter3D(pose3D[0, link], pose3D[2, link], -pose3D[1, link], color=(0,0,1))
   
        joints=argv
        for j in range(len(joints[0])):
            ax.scatter(joints[0][j][0],joints[0][j][2],-joints[0][j][1], color=(0,0,1))
        #ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('auto')

        X = pose3D[0, :]
        Y = pose3D[2, :]
        Z = -pose3D[1, :]
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        plt.title(Title, y=0.95,fontsize=25)
        plt.show()
        fig.savefig('{}.jpg'.format(fign),bbox_inches='tight')
        
    except Exception as err:
        print(type(err))
        print(err.args)
        print(err)
        sys.exit(2)
        

def rotate_pose(data, rotation_joint=1, rotation_matrix=None, m_coeff=None):
    '''
    Input: human pose keypoints, rotation joint, rotation matrix, matrix coefficient
    return: rotated human pose coordinates, rotation matrix
    '''
    if rotation_matrix is None:
        rotation_matrix = np.zeros((data.shape[0], 3, 3))

        for i in range(data.shape[0]):

            # X, Z coordinates of ankle joint
            m = np.arctan2(data[i, rotation_joint, 0], data[i, rotation_joint, 2])

            if m_coeff is not None:
                m=m_coeff

            # Rotation Matrix
            R = np.array(([np.cos(m), 0, np.sin(m)],
                          [0, 1, 0],
                          [-np.sin(m), 0, np.cos(m)]))

            rotation_matrix[i] = R

    data = np.matmul(data, rotation_matrix)
    return data, rotation_matrix

# get angle of 3 coordinates in 2D plane
def getAngle(a, b, c):
    '''
    Input: Coordinates of 3 points
    return: Angle in degree
    '''
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def directionOfPoint(Ax, Ay, Bx, By, Px, Py ):
    '''
    Input: Coordinates of 3 points
    return: cross product
    '''
    # Subtracting co-ordinates of
    # point A from B and P, to
    # make A as origin
    Bx -= Ax
    By -= Ay
    Px -= Ax
    Py -= Ay
  
    # Determining cross Product
    cross_product = Bx * Py - By * Px
#     print('cross_product: ',cross_product)
    
    return cross_product

def quad(coord):
    '''
    Determine quadrant of keypoints
    Input: Coordinate
    return: Quadrant information
    '''
    q = 0
    if coord[0] <= 0 and coord[1] <= 0:
        q = 1
    elif coord[0] >= 0 and coord[1] <= 0:
        q = 2
    elif coord[0] >= 0 and coord[1] >= 0:
        q = 3
    elif coord[0] <= 0 and coord[1] >= 0:
        q = 4
    return q

def score_c_to_5_classes(score_c):
        # type: (np.ndarray) -> int
        '''
        Score C to 5 risk-classes
        :param score_c:  Score C
        :return: Risk-class
        '''
        if score_c == 1:
            ret = 0
        elif 2 <= score_c <= 3:
            ret = 1
        elif 4 <= score_c <= 7:
            ret = 2
        elif 8 <= score_c <= 10:
            ret = 3
        else:
            ret = 4

        return ret