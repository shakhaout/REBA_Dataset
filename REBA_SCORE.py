#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import utils
import math
import argparse


# The 17 joints - in order - are:
# 
# | Pelvis | RHip | RKnee | RAnkle | LHip | LKnee | LAnkle | Spine1 | Neck | Head | Site | LShoulder | LElbow | LWrist | RShoulder | RElbow | RWrist|
# | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
# | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |



class RebaScore:
    '''
    Class to compute REBA metrics
  
    '''
    def __init__(self,pose):
        # Human pose keypoints
        self.pose = pose
        # Table A ( Neck X Trunk X Legs)
        self.table_a = np.zeros((3, 5, 4))
        # Table B ( UpperArm X LowerArm)
        self.table_b = np.zeros((6, 2))
        # Table C ( ScoreA X ScoreB)
        self.table_c = np.zeros((9, 8))

        # Body Params
        self.body = {'neck_angle': 0, 'neck_side': False,
                     'trunk_angle': 0, 'trunk_side': False,
                     'leg_score': False, 'leg_angle': 0
                     }

        # Arms Params
        self.arms = {'upper_arm_angle': 0, 'shoulder_raised': False, 'arm_abducted': False, 'leaning': False,
                     'lower_arm_angle': 0}

        # Init lookup tables
        self.init_table_a()
        self.init_table_b()
        self.init_table_c()


    def init_table_a(self):
        '''
        Table used to compute Score A
        :return: None
        '''
        self.table_a = np.array([
                                [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
                                [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
                                [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]]
                                ])

    def init_table_b(self):
        '''
        Table used to compute Score B
        :return: None
        '''
        self.table_b = np.array([
                                [1, 1],
                                [1, 2],
                                [3, 4],
                                [4, 5],
                                [6, 7],
                                [7, 8]
                                ])

    def init_table_c(self):
        '''
        Table to compute score_c
        :return: None
        '''
        self.table_c = np.array([
                                [1, 1, 1, 2, 3, 3, 4, 5],
                                [1, 2, 2, 3, 4, 4, 5, 6],
                                [2, 3, 3, 3, 4, 5, 6, 7],
                                [3, 4, 4, 4, 5, 6, 7, 8],
                                [4, 4, 4, 5, 6, 7, 8, 8],
                                [6, 6, 6, 7, 8, 8, 9, 9],
                                [7, 7, 7, 8, 9, 9, 9, 10],
                                [8, 8, 8, 9, 10, 10, 10, 10],
                                [9, 9, 9, 10, 10, 10, 11, 11]
                                ])

    def set_body(self, values):
        # type: (np.ndarray) -> None
        '''
        Set body params
        :param values: [neck_angle, neck_side, trunk_angle, trunk_side,
                        leg_score, leg_angle]
        :return: None
        '''
        assert len(values) == len(self.body)

        for i, (key, _) in enumerate(self.body.items()):
            self.body[key] = values[i]

    def set_arms(self, values):
        # type: (np.ndarray) -> None
        '''
        Set arms params
        :param values:  [upper_arm_angle, shoulder_raised, arm_abducted, leaning,
                        lower_arm_angle]
        :return: None
        '''
        assert len(values) == len(self.arms)

        for i, (key, _) in enumerate(self.arms.items()):
            self.arms[key] = values[i]

    def compute_score_a(self):
        # type: (RebaScore) -> (np.ndarray, np.ndarray)
        '''
        Compute score A
        >>> rebascore = RebaScore()
        >>> rebascore.set_body(np.array([10, 0, 20, 0, 1, 50, 0]))
        >>> rebascore.compute_score_a()
        (4, array([1, 2, 3]))
        :return: Score A, [neck_score, trunk_score, leg_score]
        '''
        neck_score, trunk_score, leg_score, load_score = 0, 0, 0, 0

        # Neck position
        if 0 <= self.body['neck_angle'] <= 20 :
            neck_score +=1
        else:
            neck_score +=2
        # Neck adjust
        neck_score +=1 if self.body['neck_side'] else 0
        neck_score = int(neck_score)
        
        # Trunk position
        if self.body['trunk_angle'] == 0:
            trunk_score +=1
        elif 0 < self.body['trunk_angle'] <= 20 or -20 <= self.body['trunk_angle'] < 0:
            trunk_score +=2
        elif 20 < self.body['trunk_angle'] <= 60 or self.body['trunk_angle'] < -20:
            trunk_score +=3
        elif self.body['trunk_angle'] > 60:
            trunk_score +=4
        # Trunk adjust
        trunk_score += 1 if self.body['trunk_side'] else 0
        trunk_score = int(trunk_score)
        
        # Legs position
        leg_score = self.body['leg_score'] 
        # Legs adjust
        if 30 <= self.body['leg_angle'] <= 60:
            leg_score += 1
        elif self.body['leg_angle'] > 60:
            leg_score += 2
        leg_score = int(leg_score)
        
        assert neck_score > 0 and trunk_score > 0 and leg_score > 0

        score_a = self.table_a[neck_score-1][trunk_score-1][leg_score-1]
        return score_a, np.array([neck_score, trunk_score, leg_score])

    def compute_score_b(self):
        # type: (RebaScore) -> (np.ndarray, np.ndarray)
        '''
        Compute score B
        >>> rebascore = RebaScore()
        >>> rebascore.set_arms(np.array([45, 0, 0, 0, 70, 0, 1]))
        >>> rebascore.compute_score_b()
        (2, array([2, 1, 2]))
        :return: scoreB, [upper_arm_score, lower_arm_score]
        '''
        upper_arm_score, lower_arm_score = 0, 0

        # Upper arm position
        if -20 <= self.arms['upper_arm_angle'] <= 20:
            upper_arm_score +=1
        elif 20 < self.arms['upper_arm_angle'] <= 45 or self.arms['upper_arm_angle'] < -20:
            upper_arm_score +=2
        elif 45 < self.arms['upper_arm_angle'] <= 90:
            upper_arm_score +=3
        elif self.arms['upper_arm_angle'] > 90:
            upper_arm_score +=4

        # Upper arm adjust
        upper_arm_score += 1 if self.arms['shoulder_raised'] else 0
        upper_arm_score += 1 if self.arms['arm_abducted'] else 0
        upper_arm_score -= 1 if self.arms['leaning'] else 0
        upper_arm_score = int(upper_arm_score)
        
        # Lower arm position
        if 60 <= self.arms['lower_arm_angle'] <= 100:
            lower_arm_score += 1
        else:
            lower_arm_score += 2
        lower_arm_score = int(lower_arm_score)
        
        assert lower_arm_score > 0 

        score_b = self.table_b[upper_arm_score-1][lower_arm_score-1]
        return score_b, np.array([upper_arm_score, lower_arm_score])
    

    def compute_score_c(self, score_a, score_b):
        # type: (np.ndarray, np.ndarray) -> (np.ndarray, str)
        '''
        Compute score C
        :param score_a:  Score A
        :param score_b:  Score B
        :return: Score C, caption
        '''
        reba_risk = ['Negligible',
                     'Low',
                     'Medium',
                     'High',
                     'Very High'
                     ]
        reba_action = ['None necessary',
                       'May be necessary',
                       'Necessary',
                       'Necessary soon',
                       'Necessary now'
                        ]

        score_c = self.table_c[score_a-1][score_b-1]
        ix = utils.score_c_to_5_classes(score_c)
        risk = reba_risk[ix]
        action = reba_action[ix]

        return score_c, risk, action

 
    
    def get_body_angles_from_pose(self, direction =None, verbose=True):
        '''
        Get body angles from pose (look at left)
        :param pose: Pose (Joints coordinates)
        :param verbose: If true show each pose for debugging
        :return: Body params (neck_angle, neck_side, trunk_angle, trunk_side,
                leg_score, legs_angle)
        '''

        pose = np.expand_dims(np.copy(self.pose), 0)

        neck_angle, neck_side, trunk_angle, trunk_side, leg_score, leg_angle = 0, 0, 0, 0, 0, 0

        if verbose:
            utils.show_skeleton(pose, title="GT pose")

        # Neck position
        N_pose, _ = utils.rotate_pose(pose, rotation_joint=1)
        neck_angle = 180 - (utils.getAngle(N_pose[0, 9][:2],N_pose[0, 8][:2],N_pose[0, 0][:2]))

        if verbose:
            utils.show_skeleton(N_pose, title="Neck angle: " + str(round(neck_angle, 2)))

        # Neck bending
        Nb_pose, _ = utils.rotate_pose(N_pose, rotation_joint=1, m_coeff=np.pi / 2)
        neck_side_angle = 180 - utils.getAngle(Nb_pose[0, 9][:2],Nb_pose[0, 8][:2],Nb_pose[0, 0][:2])
        neck_side = 1 if neck_side_angle > 10 else 0 # set the threshold here

        if verbose:
            utils.show_skeleton(Nb_pose, title="Neck side angle: " + str(round(neck_side_angle, 2)))

        # Trunk position
        T_pose, _ = utils.rotate_pose(pose, rotation_joint=1)

        if utils.quad(T_pose[0, 8]) < 3:
            trunk_angle = np.rad2deg(abs(np.arctan2(T_pose[0, 8, 1], T_pose[0, 8, 0])) - (np.pi / 2))
        else:
            trunk_angle = 270 - np.rad2deg(np.arctan2(T_pose[0, 8, 1], T_pose[0, 8, 0]))

        if verbose:
            utils.show_skeleton(T_pose, title="Trunk angle: " + str(round(trunk_angle, 2)))

        # Trunk side bending
        Tb_pose, _ = utils.rotate_pose(T_pose, rotation_joint=1, m_coeff=np.pi/2)

#         angle = np.pi /2 if utils.quad(Tb_pose[0, 8]) > 2 else -np.pi/2
        trunk_side_angle = abs(90 - abs(utils.getAngle(Tb_pose[0, 1][:2],Tb_pose[0, 0][:2],Tb_pose[0, 8][:2])))
        trunk_side = 1 if trunk_side_angle > 10 else 0

        if verbose:
            utils.show_skeleton(Tb_pose, title="Trunk side angle: " + str(round(trunk_side_angle, 2)))



        # Legs position
        L_pose, _ = utils.rotate_pose(pose, rotation_joint=1)
        leg_position = abs(abs(L_pose[0,3,1])-abs(L_pose[0,6,1]))
        leg_score = 2 if leg_position > 50 else 1 # set the threshold here
        # Leg Bending
        if direction == 'left':
            leg_angle = 180 - utils.getAngle(L_pose[0, 4][:2],L_pose[0, 5][:2],L_pose[0, 6][:2])
        elif direction == 'right':
            leg_angle = 180 - utils.getAngle(L_pose[0, 1][:2],L_pose[0, 2][:2],L_pose[0, 3][:2])
        else:
            print('Please mention either left or right direction!')

        if verbose:
            title = "Leg angle: " + str(round(leg_angle, 2)) + "  Leg score: " + str(round(leg_score, 2))
            utils.show_skeleton(L_pose, title=title)

        return np.array([neck_angle, neck_side, trunk_angle, trunk_side, leg_score, leg_angle])

    def get_arms_angles_from_pose_left(self, verbose=True):
        # type: (np.ndarray, bool) -> np.ndarray
        '''
        Get arms angles from pose (look at left)
        :param pose: Pose (Joints coordinates)
        :param verbose: If true show each pose for debugging
        :return: Body params (upper_arm_angle, shoulder_raised, arm_abducted, leaning,
                              lower_arm_angle)
        '''
        pose = np.expand_dims(np.copy(self.pose), 0)
        if verbose:
            utils.show_skeleton(pose, title="GT pose")

        # Leaning
        T_pose, _ = utils.rotate_pose(pose, rotation_joint=1)
        if utils.quad(T_pose[0, 8]) < 3:
            trunk_angle = np.rad2deg(abs(np.arctan2(T_pose[0, 8, 1], T_pose[0, 8, 0])) - (np.pi / 2))
        else:
            trunk_angle = 270 - np.rad2deg(np.arctan2(T_pose[0, 8, 1], T_pose[0, 8, 0]))
        leaning = -1 if trunk_angle > 30 else 0
        if verbose:
            utils.show_skeleton(T_pose, title="Leaning angle: " + str(round(trunk_angle, 2)))

        # Upper Arm position
        ua_pose, _ = utils.rotate_pose(pose, rotation_joint=1)
        # get the point along the upper arm parallel to spine
        p= np.array([ua_pose[0,11,0]+ua_pose[0,0,0]-ua_pose[0,8,0],ua_pose[0,11,1]+ua_pose[0,0,1]-ua_pose[0,8,1]])
        upper_arm_angle = utils.getAngle(p,ua_pose[0, 11][:2],ua_pose[0,12][:2])
        upper_arm_dir = utils.directionOfPoint(ua_pose[0,8,0],ua_pose[0,8,1],ua_pose[0,0,0],ua_pose[0,0,1],ua_pose[0,12,0],ua_pose[0,12,1])
    #     print('Upper arm direction: ',upper_arm_dir)
        upper_arm_angle = upper_arm_angle if upper_arm_dir>0 else -(upper_arm_angle)
        if verbose:
            utils.show_skeleton(ua_pose, title="Left Upper Arms angle: " + str(round(upper_arm_angle, 2)))

        # Upper Arm Adjust
        Ua_pose, _ = utils.rotate_pose(ua_pose, rotation_joint=1, m_coeff=np.pi/2)
        shoulder_step = Ua_pose[0, 8, 1] - pose[0, 11, 1]
    #     print('shoulder step: ',shoulder_step)

        # get the point along the upper arm parallel to spine
        p= np.array([Ua_pose[0,11,0]+Ua_pose[0,0,0]-Ua_pose[0,8,0],Ua_pose[0,11,1]+Ua_pose[0,0,1]-Ua_pose[0,8,1]])
        arm_abducted_angle = utils.getAngle(p,Ua_pose[0, 11][:2],Ua_pose[0,12][:2])
        upper_arm_abd_dir = utils.directionOfPoint(Ua_pose[0,11,0],Ua_pose[0,11,1],p[0],p[1],Ua_pose[0,12,0],Ua_pose[0,12,1])
    #     print('Upper arm direction: ',upper_arm_abd_dir)
        arm_abducted_angle = arm_abducted_angle if upper_arm_abd_dir<0 else -(arm_abducted_angle)

        shoulder_raised = 1 if shoulder_step > 1 else 0 # set exact threshold
        arm_abducted = 1 if arm_abducted_angle > 30 else 0 # set exact threshold

        if verbose:
            utils.show_skeleton(Ua_pose, title="Left Upper Arms abducted: " + str(round(arm_abducted_angle, 2)))

        # Lower Arm position
        la_pose, _ = utils.rotate_pose(pose, rotation_joint=1)
        lower_arm_angle = abs(180 - utils.getAngle(la_pose[0, 13][:2],la_pose[0, 12][:2],la_pose[0, 11][:2]))

        if verbose:
            utils.show_skeleton(la_pose, title="Left Lower Arms angle: " + str(round(lower_arm_angle, 2)))


        return np.array([upper_arm_angle, shoulder_raised, arm_abducted, leaning,
                lower_arm_angle])

    def get_arms_angles_from_pose_right(self, verbose=True):
        # type: (np.ndarray, bool) -> np.ndarray
        '''
        Get arms angles from pose (look at right)
        :param pose: Pose (Joints coordinates)
        :param verbose: If true show each pose for debugging
        :return: Body params (upper_arm_angle, shoulder_raised, arm_abducted, leaning,
                              lower_arm_angle)
        '''
        pose = np.expand_dims(np.copy(self.pose), 0)
        if verbose:
            utils.show_skeleton(pose, title="GT pose")

        # Leaning
        T_pose, _ = utils.rotate_pose(pose, rotation_joint=1)
        if utils.quad(T_pose[0, 8]) < 3:
            trunk_angle = np.rad2deg(abs(np.arctan2(T_pose[0, 8, 1], T_pose[0, 8, 0])) - (np.pi / 2))
        else:
            trunk_angle = 270 - np.rad2deg(np.arctan2(T_pose[0, 8, 1], T_pose[0, 8, 0]))
        leaning = -1 if trunk_angle > 30 else 0
        if verbose:
            utils.show_skeleton(T_pose, title="Leaning angle: " + str(round(trunk_angle, 2)))

        # Upper Arm position
        ua_pose, _ = utils.rotate_pose(pose, rotation_joint=1)
        # get the point along the upper arm parallel to spine
        p= np.array([ua_pose[0,14,0]+ua_pose[0,0,0]-ua_pose[0,8,0],ua_pose[0,14,1]+ua_pose[0,0,1]-ua_pose[0,8,1]])
        upper_arm_angle = utils.getAngle(p,ua_pose[0, 14][:2],ua_pose[0,15][:2])
        upper_arm_dir = utils.directionOfPoint(ua_pose[0,8,0],ua_pose[0,8,1],ua_pose[0,0,0],ua_pose[0,0,1],ua_pose[0,15,0],ua_pose[0,15,1])
    #     print('Upper arm direction: ',upper_arm_dir)
        upper_arm_angle = upper_arm_angle if upper_arm_dir>0 else -(upper_arm_angle)
        if verbose:
            utils.show_skeleton(ua_pose, title="Right Upper Arms angle: " + str(round(upper_arm_angle, 2)))

        # Upper Arm Adjust
        Ua_pose, _ = utils.rotate_pose(ua_pose, rotation_joint=1, m_coeff=np.pi/2)
        shoulder_step = Ua_pose[0, 8, 1] - pose[0, 14, 1]
    #     print('shoulder step: ',shoulder_step)

        # get the point along the upper arm parallel to spine
        p= np.array([Ua_pose[0,14,0]+Ua_pose[0,0,0]-Ua_pose[0,8,0],Ua_pose[0,14,1]+Ua_pose[0,0,1]-Ua_pose[0,8,1]])
        arm_abducted_angle = utils.getAngle(p,Ua_pose[0, 14][:2],Ua_pose[0,15][:2])
        upper_arm_abd_dir = utils.directionOfPoint(Ua_pose[0,14,0],Ua_pose[0,14,1],p[0],p[1],Ua_pose[0,15,0],Ua_pose[0,15,1])
    #     print('Upper arm direction: ',upper_arm_abd_dir)
        arm_abducted_angle = arm_abducted_angle if upper_arm_abd_dir>0 else -(arm_abducted_angle)

        shoulder_raised = 1 if shoulder_step > 1 else 0 # set exact threshold
        arm_abducted = 1 if arm_abducted_angle > 30 else 0 # set exact threshold

        if verbose:
            utils.show_skeleton(Ua_pose, title="Right Upper Arms abducted: " + str(round(arm_abducted_angle, 2)))

        # Lower Arm position
        la_pose, _ = utils.rotate_pose(pose, rotation_joint=1)
        lower_arm_angle = abs(180 - utils.getAngle(la_pose[0, 16][:2],la_pose[0, 15][:2],la_pose[0, 14][:2]))

        if verbose:
            utils.show_skeleton(la_pose, title="Right Lower Arms angle: " + str(round(lower_arm_angle, 2)))


        return np.array([upper_arm_angle, shoulder_raised, arm_abducted, leaning,
                lower_arm_angle])
    


if __name__ == '__main__':
    
    # User input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-kpt',  help='Input the Human Pose Keypoints Array')
    parser.add_argument('-direction',  help='Mention left or right')
    args= parser.parse_args()

    sample_pose = args.kpt
    rebaScore = RebaScore(sample_pose)
    
    if args.direction == 'left':
        body_params = rebaScore.get_body_angles_from_pose('left')
        arms_params = rebaScore.get_arms_angles_from_pose_left()
    elif args.direction == 'right':
        body_params = rebaScore.get_body_angles_from_pose('right')
        arms_params = rebaScore.get_arms_angles_from_pose_right()
    else:
        print('Input the left/right direction to calculate REBA score!')
        
    rebaScore.set_body(body_params)
    score_a, partial_a = rebaScore.compute_score_a()
    rebaScore.set_arms(arms_params)
    score_b, partial_b = rebaScore.compute_score_b()
    score_c, risk, action = rebaScore.compute_score_c(score_a, score_b)
    print("Score A: ", score_a, "Neck Score: ", partial_a[0],"Trunk Score: ",partial_a[1],"Leg Score: ",partial_a[2])
    print("Score B: ", score_b, "Upper Arm Score: ", partial_b[0],"Lower Arm Score: ",partial_b[1])
    print("Score C: ", score_c, "Risk: ", risk, "Action: ",action)
        
