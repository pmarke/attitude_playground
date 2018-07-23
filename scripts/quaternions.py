import numpy as np 


class Quaternion:

    # Constructs the quaternion q = [w,x,y,z] where
    # w is the scalar and x,y,z form the vector.
    def __init__(self, *args):

        self._w, self._x, self._y, self._z = 1,0,0,0

        if len(args) == 0: # init identity quaternion
            self.init_quaternion(1,0,0,0)
        elif len(args) == 1: # init quaternion from rotation matrix
            self.r_to_q(args[0])
        elif len(args) == 2: # init quaternion from axel angel
            self.aa_to_q(args[0],args[1]) 
        elif len(args) == 3: # init quaternion from euler angles 
            self.euler_to_q(args[0],args[1],args[2])
        elif len(args) == 4: # init quaternion from scalar and vector
            self.init_quaternion(args[0],args[1],args[2],args[3])
        else:
            print('Input argument not understood. Using default')
            self.init_quaternion([1,0,0,0])



    # Basic quaternion constructor
    def init_quaternion(self, w, x, y, z):
        self._w = w     
        self._x = x     
        self._y = y     
        self._z = z  



    # Converts the rotation matrix to quaternion. 
    def r_to_q(self, R = np.identity(3)):
        self._w = 0.5*np.sqrt(1+np.float(R.trace()))
        self._x = 0.5*np.sqrt(1+R[0,0]-R[1,1]-R[2,2])
        self._y = 0.5*np.sqrt(1-R[0,0]+R[1,1]-R[2,2])
        self._z = 0.5*np.sqrt(1-R[0,0]-R[1,1]+R[2,2])



    # Converts quaternion to rotation matrix
    def q_to_r(self):

        r11 = self._w**2+self._x**2-self._y**2-self._z**2
        r12 = 2*(self._x*self._y + self._w*self._z)
        r13 = 2*(self._x*self._z - self._w*self._y)

        r21 = 2*(self._x*self._y - self._w*self._z)
        r22 = self._w**2-self._x**2+self._y**2-self._z**2
        r23 = 2*(self._y*self._z + self._w*self._x)

        r31 = 2*(self._x*self._z + self._w*self._y)
        r32 = 2*(self._y*self._z - self._w*self._x)
        r33 = self._w**2-self._x**2-self._y**2+self._z**2

        R = np.matrix([[r11,r12,r13], \
                      [r21,r22,r23], \
                      [r31,r32,r33]])

        return R



    # Converts angle axis to quaternion.
    # - theta is the angle of rotation in radians
    # - n is the axes about rotation 
    def aa_to_q(self, theta=0, n=np.array([0,0,0])):
        self._w = np.cos(theta/2)
        self._x = np.sin(theta/2)*n[0]
        self._y = np.sin(theta/2)*n[1]
        self._z = np.sin(theta/2)*n[2]



    # Converts quaternion to angle axis
    def q_to_aa(self):

        theta = 2*np.arcsin(np.sqrt(self._x**2+self._y**2+self._z**2))

        # There is no rotation, return identity element ?????
        if np.round(theta,4) == 0:
            theta = 0
            n1,n2,n3 = 0,0,0
        # There is rotation, compute vector elements
        else:
            n1 = self._x/np.sin(theta/2)
            n2 = self._y/np.sin(theta/2)
            n3 = self._z/np.sin(theta/2)

        return theta, np.array([n1,n2,n3])



    # Converts 321 sequence of euler angles to quaternion
    def euler_to_q(self, phi=0, theta=0, psi=0):
        c_psi,c_th,c_phi,s_psi,s_th,s_phi = np.cos(psi/2),np.cos(theta/2),np.cos(phi/2),np.sin(psi/2),np.sin(theta/2),np.sin(phi/2)
        self._w = c_psi*c_th*c_phi + s_psi*s_th*s_phi
        self._x = c_psi*c_th*s_phi - s_psi*s_th*c_phi
        self._y = c_psi*s_th*c_phi + s_psi*c_th*s_phi
        self._z = s_psi*c_th*c_phi - c_psi*s_th*s_phi



    # Returns the 321 sequence of euler angels
    def q_to_euler(self):
        theta = np.arcsin(2*(self._w*self._y-self._z*self._x))
        if (np.round(np.cos(theta),4) != 0):
            phi = np.arctan2(self._x+self._z,self._w-self._y) + np.arctan2(self._x-self._z,self._w+self._y)
            psi = np.arctan2(self._x+self._z,self._w-self._y) - np.arctan2(self._x-self._z,self._w+self._y)
        else:
            psi = 0
            if np.round(theta,4) == np.round(np.pi/2,4):
                phi = np.arctan2(self._x-self._z,self._w-self._y)
            else:
                phi = np.arctan2(self._x+self._z,self._w+self._y)
        return phi,theta,psi



    # Sets the quaternion to another quaternion
    def equal(self,q):
        self._w, self._x, self._y, self._z = q._w, q._x, q._y, q._z



    # Returns a copy of itself
    def copy(self):
        q = Quaternion()
        q.equal(self)
        return q



    # Returns the identity group element of the quaternion
    @staticmethod
    def identity():
        return Quaternion(1,0,0,0)



    # Overrides the addition operation
    def __add__(self,other):
        if other.__class__.__name__ == self.__class__.__name__:
            q = Quaternion(self._w+other._w, self._x+other._x, self._y+other._y, self._z+other._z)
        else:
            print("Only can do addition with type: "+self.__class__.__name__)
            q = Quaternion()
        return q



    # Implements quaternion division by a scalar
    def __div__(self,other):
        if np.isscalar(other):
            return self.scalar_mult(self,1.0/other)
        else:
            print("Only can do division with scalar")



    # Overrides the multiplication operation
    def __mul__(self,other):
        if other.__class__.__name__ == self.__class__.__name__:
            return self.quaternion_mult(self, other)
        elif isinstance(other,np.ndarray):
            return self.vector_rot(self,other)
        elif np.isscalar(other):
            return self.scalar_mult(self,other)
        else:
            print("Multiplication with type: "+ type(other).__name__ +" not implemented.")


        

    # Implements quaternion multiplication by a scalar
    @staticmethod
    def scalar_mult(q,s):
        return Quaternion(q._w*s,q._x*s,q._y*s,q._z*s)



    # Multiplies two quaternions together q1*q2
    @staticmethod
    def quaternion_mult(q2,q1):
        # q2.p()
        n_prime = q2.mat()
        n = np.matrix([[q1._x],[q1._y],[q1._z],[q1._w]])
        n_dprime = n_prime*n
        return Quaternion(n_dprime[3,0],n_dprime[0,0],n_dprime[1,0],n_dprime[2,0])



    # Rotates a vector by the quaternion
    @staticmethod
    def vector_rot(q,v):
        V = Quaternion(0,v[0,0],v[1,0],v[2,0]) # convert vector to quaternion
        V_prime = q*V*q.inv()
        return np.array([[V_prime._x],[V_prime._y],[V_prime._z]])



    # Converts the quaternion into matrix representation
    def mat(self):
        return np.matrix([[ self._w,  self._z, -self._y, self._x],
                          [-self._z,  self._w,  self._x, self._y],
                          [ self._y, -self._x,  self._w, self._z],
                          [-self._x, -self._y, -self._z, self._w]])



    # Returns the inverse of the quaternion
    def inv(self):
        return Quaternion(self._w, -self._x, -self._y, -self._z)



    # Returns the vector component of the quaternion
    def n(self):
        return np.array([[self._x],[self._y],[self._z]])



    # Transformation from algebra to group
    # - delta: 3d vector representing the axis-angle rotation
    #   ex: delta = np.array([[wx],[wy],[wz]])
    @staticmethod
    def from_algebra(delta):
        delta_norm = np.linalg.norm(delta)

        if (delta_norm > 1e-10):
            w = np.cos(delta_norm/2)
            n = np.sin(delta_norm/2)*delta/delta_norm
        else:
            w = 1
            n = delta/2
        return Quaternion(w,n[0,0],n[1,0],n[2,0])



    # Transformation from group to algebra
    # - delta: 3d vector representing the axis-angle rotation
    #   ex: delta = np.array([[wx],[wy],[wz]])
    def to_algebra(self):
        n_norm = np.linalg.norm(self.n())

        if (n_norm > 1e-10):
            delta = 2*np.arctan2(n_norm,self._w)*self.n()/n_norm
        else:
            delta = np.sign(self._w)*self.n()
        return delta

        

    # Normalize the quaternion so that
    # sqrt(w**2 + x**2 + y**2 + z**2) = 1
    def normalize(self):
        n = np.array([self._w, self._x, self._y, self._z])
        n = n/np.linalg.norm(n)
        self._w, self._x, self._y, self._z = n[0],n[1],n[2],n[3]

    # def get_derivative(self,w,Ts):

    #     mat = np.matrix([[ self._w, -self._z,  self._y],
    #                      [ self._z,  self._w, -self._x],
    #                      [-self._y,  self._x,  self._w],
    #                      [-self._x, -self._y, -self._z]])

    #     n = 0.5*mat*w*Ts
    #     # print(mat)
    #     # print(w)
    #     # print(n)
    #     return Quaternion(np.float(n[3]),np.float(n[0]),np.float(n[1]),np.float(n[2]))

#-------------------------------------------------------------------
#                Used for Debugging
#------------------------------------------------------------------

    # Compares quaternion with another to see if they are the same 
    def compare(self, w,x,y,z):
        dp = 5
        return (np.round(self._w,dp) == np.round(w,dp)) and (np.round(self._x,dp) == np.round(x,dp)) and (np.round(self._y,dp) == np.round(y,dp)) and (np.round(self._z,dp) == np.round(z,dp))

    # Prints Quaternion elements
    def p(self):
        print(str(self._w),"("+str(self._x)+","+str(self._y)+","+str(self._z)+")")


#-------------------------------------------------------------------
#              Add derivative method
#------------------------------------------------------------------

class Quaternion(Quaternion):

    # Returns the derivative of the quaternion
    def get_derivative(self,w,Ts):

        mat = np.matrix([[ self._w, -self._z,  self._y],
                         [ self._z,  self._w, -self._x],
                         [-self._y,  self._x,  self._w],
                         [-self._x, -self._y, -self._z]])

        n = 0.5*mat*w*Ts
        # print(mat)
        # print(w)
        # print(n)
        return Quaternion(np.float(n[3]),np.float(n[0]),np.float(n[1]),np.float(n[2]))