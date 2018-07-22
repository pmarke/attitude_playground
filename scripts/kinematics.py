import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from quaternions import Quaternion


#############################################################
#             Generate Gyro Samples
#############################################################

def generate_gyro_samples(w0=np.array([[50],[30],[20]]), Ts=0.01, secs=5, alpha=0.1, sigma=0.25):
    
    # Number of samples
    N = np.int(secs/Ts)

    # Sample time vector
    n=np.arange(N)

    # Generate the slowly-decaying signal
    w=np.multiply(w0,np.exp(-alpha*n*Ts))

    # Make some noise!
    eta = sigma*np.random.randn(N)

    d = {'w':w+eta*0, 'Ts':Ts, 'secs':secs, 'alpha':alpha, 'sigma':sigma}

    return d

# samples = generate_gyro_samples()
# w = samples['w']
# fig = plt.figure()
# plt.plot(w.transpose())
# plt.title('Gyro Samples over time')
# plt.grid();plt.xlabel('Sample'); plt.ylabel('Angular Velocity [rad/s]')
# plt.show()

#############################################################
#             Show Visualization Samples
#############################################################

class Visualize:

    def __init__(self,v):

        self.v = v

        # Create the figure
        self.fig = plt.figure(figsize=(7,7))
        self.ax = self.fig.add_subplot(111,projection='3d')

        # Create the manifold
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the manifold
        self.ax.plot_surface(x, y, z, color='b', alpha = 0.2)
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.set_zlabel("Z-axis")
        plt.title("Attitude of object")

    # Rotates the vector by every quaternion in the list trajectory
    # If trajectory is not provided, the vector will be rotated by the
    # identity quaternion
    def draw(self,trajectory=[Quaternion()]):

        for i in range(len(trajectory)):
            trajectory[i].p()
            v = trajectory[i]*self.v

            # print(v)
            self.ax.scatter(v[0,0],v[1,0],v[2,0],s=50,c='r', edgecolors='none')


# n = np.array([[1],[3],[4]])
# n = n/np.linalg.norm(n)
# viz = Visualize(n)
# viz.draw()
# plt.show()


#############################################################
#                Euler Integration
#############################################################


class Quaternion(Quaternion):

    def euler_integration(self, w, Ts=0.01, normalize=False):

        # Store trajectory
        trajectory = [Quaternion()]

        # Perform Euler integration for every gyro sample
        for i in range(w.shape[1]):
            self.equal(self+self.get_derivative(w[:,i].reshape(3,1),Ts))
            
            if normalize:
                self.normalize()
            # self.p()
            trajectory.append(self.copy())

        return trajectory


# # Create Quaternion and visualization objects
# n = np.array([[1],[0],[0]])
# n = n/np.linalg.norm(n)
# viz = Visualize(n)
# q = Quaternion()

# # Generate gyro samples and create trajectory
# samples = generate_gyro_samples(w0=np.array([[50],[50],[0]]), Ts=0.01, secs=0.1, alpha=0.1, sigma=0.25)
# trajectory = q.euler_integration(samples['w'],samples['Ts'])

# viz.draw(trajectory)
# plt.show()



#############################################################
#                Runge-Kutta 4 (RK4)
#############################################################

class Quaternion(Quaternion):

    def rk4_integration(self,w,Ts=0.01, normalize = False):
                # Store trajectory
        trajectory = [Quaternion()]

        f = lambda i,t,q: q.get_derivative(c,1)

        # Perform Euler integration for every gyro sample
        for i in range(w.shape[1]):

            t = i*Ts
            k1 = f(i,t,self)
            # k1.p()
            k2 = f(i, t+Ts/2, self + k1*(Ts/2))
            k3 = f(i, t+Ts/2, self + k2*(Ts/2))
            k4 = f(i, t+Ts,   self + k3*Ts)
            self += (k1+k2*2+k3*2+k4)*(Ts/6)
            
            if normalize:
                self.normalize()
            # self.p()
            trajectory.append(self.copy())

        return trajectory

# # Create Quaternion and visualization objects
# n = np.array([[1],[0],[0]])
# n = n/np.linalg.norm(n)
# viz = Visualize(n)
# q = Quaternion()

# # Generate gyro samples and create trajectory
# samples = generate_gyro_samples(w0=np.array([[50],[50],[0]]), Ts=0.01, secs=1, alpha=0.1, sigma=0.25)
# trajectory = q.rk4_integration(samples['w'],samples['Ts'])

# viz.draw(trajectory)
# plt.show()

#############################################################
#                 Integrating on the Manifold
#############################################################

class Quaternion(Quaternion):

    def lie_integration(self,w,Ts):

        # Store trajectory
        trajectory = [Quaternion()]

        # Perform Euler integration for every gyro sample
        for i in range(w.shape[1]):

            q_dt = Quaternion.from_algebra(w[:,i].reshape(3,1)*Ts)
            self.equal(q_dt*self)
       
            trajectory.append(self.copy())

        return trajectory

# Create Quaternion and visualization objects
n = np.array([[1],[0],[0]])
n = n/np.linalg.norm(n)
viz = Visualize(n)
q = Quaternion()

# Generate gyro samples and create trajectory
samples = generate_gyro_samples(w0=np.array([[50],[0],[-50]]), Ts=0.01, secs=1, alpha=0.1, sigma=0.25)
trajectory = q.lie_integration(samples['w'],samples['Ts'])

viz.draw(trajectory)
plt.show()
