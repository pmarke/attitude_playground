import numpy as np 
import matplotlib.pyplot as plt 
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

    return w+eta

w = generate_gyro_samples()
fig = plt.figure()
plt.plot(w.transpose()
plt.title('Gyro Samples over time')
plt.grid();plt.xlabel('Sample'); plt.ylabel('Angular Velocity [rad/s]')
plt.show()