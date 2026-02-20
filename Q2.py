import numpy as np

class Joint:
    def __init__(self, Link_twist, Link_length, Link_offset, Joint_angle):
        self.Link_twist = np.radians(Link_twist)
        self.Link_length = Link_length
        self.Link_offset = Link_offset
        self.Joint_angle = np.radians(Joint_angle)

    def trans_matrix(self):
        matrix = np.array([
            [np.cos(self.Joint_angle), -np.sin(self.Joint_angle), 0, self.Link_length],
            [np.sin(self.Joint_angle)*np.cos(self.Link_twist), np.cos(self.Joint_angle)*np.cos(self.Link_twist), -np.sin(self.Link_twist), -self.Link_offset*np.sin(self.Link_twist)],
            [np.sin(self.Joint_angle)*np.sin(self.Link_twist), np.cos(self.Joint_angle)*np.sin(self.Link_twist), np.cos(self.Link_twist), self.Link_offset*np.cos(self.Link_twist)],
            [0, 0, 0, 1]
        ])
        print(matrix)
        return matrix

inputs = input("Please input the angle")
joint_angle = list(map(float, inputs.split()))
joint_angle[1] -= 90

shoulder_pan = Joint(0, 0, 0.0624, joint_angle[0])
shoulder_lift = Joint(-90, 0.035, 0, joint_angle[1])
elbow_flex = Joint(0, 0.116, 0, joint_angle[2])
wrist_flex = Joint(0, 0.135, 0, joint_angle[3])
wrist_roll = Joint(-90, 0, 0.061, joint_angle[4] )

shoulder_pan_trans = shoulder_pan.trans_matrix()
shoulder_lift_trans = shoulder_lift.trans_matrix()
elbow_flex_trans = elbow_flex.trans_matrix()
wrist_flex_trans = wrist_flex.trans_matrix()
wrist_roll_trans = wrist_roll.trans_matrix()

final_matrix = shoulder_pan_trans@shoulder_lift_trans@elbow_flex_trans@wrist_flex_trans@wrist_roll_trans
print(final_matrix)