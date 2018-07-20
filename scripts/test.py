import numpy as np 
from quaternions import Quaternion 

dv = 7 # decimal place to round to

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


#####################################################
#          Testing Euler Angle Conversion
#####################################################
# Using 321 sequence euler angle

print bcolors.HEADER + "Starting Euler Angle Conversion Test" + bcolors.ENDC

test = []

# Test1: No rotation
roll, pitch, yaw = np.float(0),np.float(0),np.float(0)
q = Quaternion(roll,pitch,yaw)
test.append(q.compare(1,0,0,0))

# Test2: Rotation about the z axis
roll, pitch, yaw = np.float(0),np.float(0),np.float(0.0)
q = Quaternion(roll,pitch,yaw)
test.append(q.compare(np.cos(yaw/2),0,0,np.sin(yaw/2)))

# Test3: Rotation about the y axis
roll, pitch, yaw = np.float(0),np.float(0.0),np.float(0)
q = Quaternion(roll,pitch,yaw)
test.append(q.compare(np.cos(pitch/2),0,np.sin(pitch/2),0))

# Test4: Rotation about the x axis
roll, pitch, yaw = np.float(0.0),np.float(0),np.float(0)
q = Quaternion(roll,pitch,yaw)
test.append(q.compare(np.cos(roll/2),np.sin(roll/2),0,0))

# Test5: Rotation about multiple axes
roll,pitch,yaw = np.float(0.3),np.float(0.2),np.float(0.1)
q = Quaternion(roll,pitch,yaw)
roll_t, pitch_t, yaw_t = q.q_to_euler()
test.append(roll == np.round(roll_t,dv) and pitch == np.round(pitch_t,dv) and yaw == np.round(yaw_t,dv))

if (np.array(test).sum() == len(test)):
    print bcolors.OKGREEN + "All Euler Angle Conversion Tests Passed!!" + bcolors.ENDC
else:
    for i in range(len(test)):
        if test[i] == False:
            print bcolors.FAIL + "Test"+str(i+1)+" has failed!" + bcolors.ENDC
    exit()


#####################################################
#          Testing Rotation Matrix Conversion
#####################################################
# Using 321 sequence euler angle

# Define euler to rotation matrices to use
def Rotx(theta):
    Rx = np.matrix([[1,0,0],
                    [0, np.cos(theta), np.sin(theta)],
                    [0, -np.sin(theta), np.cos(theta)]])
    return Rx

def Roty(theta):
    Ry = np.matrix([[np.cos(theta), 0, -np.sin(theta)],
                    [0,1,0],
                    [np.sin(theta), 0, np.cos(theta)]])
    return Ry

def Rotz(theta):
    Rz = np.matrix([[np.cos(theta), np.sin(theta),0], 
                    [-np.sin(theta),np.cos(theta),0],
                    [0,0,1]])
    return Rz

def Rot(phi,theta,psi):
    return Rotx(phi)*Roty(theta)*Rotz(psi)


# Start tests
print bcolors.HEADER + "Starting Rotation Matrix Conversion Test" + bcolors.ENDC

test = []

# Test1: No rotation
roll, pitch, yaw = np.float(0),np.float(0),np.float(0)
R = Rot(roll,pitch,yaw)
q = Quaternion(R)
test.append(q.compare(1,0,0,0))

# Test2: Rotation about the z axis
roll, pitch, yaw = np.float(0),np.float(0),np.float(0.0)
R = Rot(roll,pitch,yaw)
q = Quaternion(R)
test.append(q.compare(np.cos(yaw/2),0,0,np.sin(yaw/2)))

# Test3: Rotation about the y axis
roll, pitch, yaw = np.float(0),np.float(0.0),np.float(0)
R = Rot(roll,pitch,yaw)
q = Quaternion(R)
test.append(q.compare(np.cos(pitch/2),0,np.sin(pitch/2),0))

# Test4: Rotation about the x axis
roll, pitch, yaw = np.float(0.0),np.float(0),np.float(0)
R = Rot(roll,pitch,yaw)
q = Quaternion(R)
test.append(q.compare(np.cos(roll/2),np.sin(roll/2),0,0))

# Test5: Rotation about multiple axes
roll,pitch,yaw = 0.4,0.2,0.1
R = Rot(roll,pitch,yaw)
q = Quaternion(R)
R_t = q.q_to_r()
sum_t = np.float((R*np.linalg.inv(R_t)).trace())
test.append(np.round(sum_t,dv) == 3.0)

if (np.array(test).sum() == len(test)):
    print bcolors.OKGREEN + "All Rotation Matrix Conversion Tests Passed!!" + bcolors.ENDC
else:
    for i in range(len(test)):
        if test[i] == False:
            print bcolors.FAIL + "Test"+str(i+1)+" has failed!" + bcolors.ENDC
    exit()


#####################################################
#          Testing Axis Angle Conversion
#####################################################

print bcolors.HEADER + "Starting Angle Axis Conversion Test" + bcolors.ENDC

test = []

# Test1: No rotation
theta = 0
n = np.array([0,0,0])
q = Quaternion(theta,n)
test.append(q.compare(1,0,0,0))

# Test2: Rotation about the z axis
yaw = 0.5
n = np.array([0,0,1])
q = Quaternion(yaw,n)
test.append(q.compare(np.cos(yaw/2),0,0,np.sin(yaw/2)))

# Test3: Rotation about the y axis
pitch = 0.5
n = np.array([0,1,0])
q = Quaternion(pitch,n)
test.append(q.compare(np.cos(pitch/2),0,np.sin(pitch/2),0))

# Test4: Rotation about the x axis
roll = 0.5
n = np.array([1,0,0])
q = Quaternion(roll,n)
test.append(q.compare(np.cos(roll/2),np.sin(roll/2),0,0))

# Test5: Rotation about multiple axes
theta = 0.5
n = np.array([1,5,-2])
n = n/np.linalg.norm(n)
q = Quaternion(theta,n)
theta_t,n_t = q.q_to_aa()
test.append(np.round(theta_t,dv) == theta and np.round(np.dot(n,n_t),dv) == 1.0)

if (np.array(test).sum() == len(test)):
    print bcolors.OKGREEN + "All Angle Axis Conversion Tests Passed!!" + bcolors.ENDC
else:
    for i in range(len(test)):
        if test[i] == False:
            print bcolors.FAIL + "Test"+str(i+1)+" has failed!" + bcolors.ENDC
    exit()

#####################################################
#        Testing Quaternion Multiplication 
#####################################################

print bcolors.HEADER + "Starting Quaternion Multiplication Test" + bcolors.ENDC

test = []

# Test1: No rotation
roll_1, pitch_1, yaw_1 = 0,0,0
roll_2, pitch_2, yaw_2 = 0,0,0
q1 = Quaternion(roll_1, pitch_1,yaw_1)
q2 = Quaternion(roll_2, pitch_2,yaw_2)
q3 = q2*q1
test.append(q3.compare(1,0,0,0))

# Test2: Rotation about the z axis
roll_1, pitch_1, yaw_1 = 0,0,0.2
roll_2, pitch_2, yaw_2 = 0,0,0.3
q1 = Quaternion(roll_1, pitch_1,yaw_1)
q2 = Quaternion(roll_2, pitch_2,yaw_2)
q3 = q2*q1
test.append(q3.compare(np.cos( (yaw_1+yaw_2)/2 ),0,0,np.sin( (yaw_1+yaw_2)/2 )))

# Test3: Rotation about the y axis
roll_1, pitch_1, yaw_1 = 0,0.2,0
roll_2, pitch_2, yaw_2 = 0,0.3,0
q1 = Quaternion(roll_1, pitch_1,yaw_1)
q2 = Quaternion(roll_2, pitch_2,yaw_2)
q3 = q2*q1
test.append(q3.compare(np.cos( (pitch_1+pitch_2)/2 ),0,np.sin( (pitch_1+pitch_2)/2 ),0))

# Test4: Rotation about the x axis
roll_1, pitch_1, yaw_1 = 0.2,0,0
roll_2, pitch_2, yaw_2 = 0.3,0,0
q1 = Quaternion(roll_1, pitch_1,yaw_1)
q2 = Quaternion(roll_2, pitch_2,yaw_2)
q3 = q2*q1
test.append(q3.compare(np.cos( (roll_1+roll_2)/2 ),np.sin( (roll_1+roll_2)/2 ),0,0))

# Test5: Rotation about multiple axis
roll_1, pitch_1, yaw_1 = 0.2,0.4,0.2
roll_2, pitch_2, yaw_2 = 0.3,0.2,0.1
q1 = Quaternion(roll_1, pitch_1,yaw_1)
q2 = Quaternion(roll_2, pitch_2,yaw_2)
q3 = q2*q1
R1 = Rot(roll_1,pitch_1,yaw_1)
R2 = Rot(roll_2,pitch_2,yaw_2)
R3 = R2*R1
R_t = q3.q_to_r()
sum_t = np.float((R3*np.linalg.inv(R_t)).trace())
test.append(np.round(sum_t,dv) == 3.0)

if (np.array(test).sum() == len(test)):
    print bcolors.OKGREEN + "All Quaternion Multiplication Tests Passed!!" + bcolors.ENDC
else:
    for i in range(len(test)):
        if test[i] == False:
            print bcolors.FAIL + "Test"+str(i+1)+" has failed!" + bcolors.ENDC
    exit()


#####################################################
#        Testing Quaternion Vector Multiplication 
#####################################################

print bcolors.HEADER + "Starting Quaternion Vector Multiplication Test" + bcolors.ENDC

test = []

# Test1: No rotation
roll, pitch, yaw = 0,0,0
q = Quaternion(roll, pitch,yaw)
R = Rot(roll,pitch,yaw)
v = np.array([[1],[2],[0.3]])
v_R = R*v
v_q = q*v
test.append(np.round(np.sum(np.cross(v_q,v_R,axis=0)**2),dv)==0)

# Test2: Rotation about the z axis
roll, pitch, yaw = 0,0,0.2
q = Quaternion(roll, pitch,yaw)
R = Rot(roll,pitch,yaw)
v = np.array([[1],[2],[0.3]])
v_R = R*v
v_q = q*v
test.append(np.round(np.sum(np.cross(v_q,v_R,axis=0)**2),dv)==0)

# Test3: Rotation about the y axis
roll, pitch, yaw = 0,0.2,0
q = Quaternion(roll, pitch,yaw)
R = Rot(roll,pitch,yaw)
v = np.array([[1],[2],[0.3]])
v_R = R*v
v_q = q*v
test.append(np.round(np.sum(np.cross(v_q,v_R,axis=0)**2),dv)==0)

# Test4: Rotation about the z axis
roll, pitch, yaw = 0.2,0,0
q = Quaternion(roll, pitch,yaw)
R = Rot(roll,pitch,yaw)
v = np.array([[1],[2],[0.3]])
v_R = R*v
v_q = q*v
test.append(np.round(np.sum(np.cross(v_q,v_R,axis=0)**2),dv)==0)

# Test5: Multiple rotations
roll, pitch, yaw = 0.4,0.2,0.1
q = Quaternion(roll, pitch,yaw)
R = Rot(roll,pitch,yaw)
v = np.array([[1],[2],[0.3]])
v_R = R*v
v_q = q*v
test.append(np.round(np.sum(np.cross(v_q,v_R,axis=0)**2),dv)==0)

if (np.array(test).sum() == len(test)):
    print bcolors.OKGREEN + "All Quaternion Vector Multiplication Tests Passed!!" + bcolors.ENDC
else:
    for i in range(len(test)):
        if test[i] == False:
            print bcolors.FAIL + "Test"+str(i+1)+" has failed!" + bcolors.ENDC
    exit()

#####################################################
#        Testing Algebra and Group Conversion
#####################################################

print bcolors.HEADER + "Starting Algebra and Group Conversion Test" + bcolors.ENDC

test = []
num_tests = 30
delta_list = []
delta_t_list = []

for i in range(num_tests):
    delta = np.random.ranf([3,1])*0.1
    q = Quaternion().from_algebra(delta)
    delta_t = q.to_algebra()
    test.append(np.round(np.sum(np.cross(delta,delta_t,axis=0)**2),dv)==0)
    delta_list.append(delta)
    delta_t_list.append(delta_t)

if (np.array(test).sum() == len(test)):
    print bcolors.OKGREEN + "All Algebra and Group Conversion Tests Passed!!" + bcolors.ENDC
else:
    for i in range(len(test)):
        if test[i] == False:
            print bcolors.FAIL + "Test"+str(i+1)+" has failed!" + bcolors.ENDC
            print "delta: \n" + str(delta_list[i]) 
            print "delta_t: \n" + str(delta_t_list[i]) + "\n"
    exit()



