import numpy as np
import pickle as pkl
import cv2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

with open("abhay3", 'rb') as fp:
    cordinates = np.array(pkl.load(fp))



print(cordinates.shape)
nFrames, nFeatures = cordinates.shape[:2]
#nFrames = 5
W = np.zeros((2*nFrames, nFeatures))
cordinates = cordinates[:nFrames,:].reshape(nFrames, nFeatures, 2)
W[0:nFrames, :] = cordinates[:nFrames,:,0]
W[nFrames:2*nFrames, :] = cordinates[:nFrames,:,1]

w_bar  = W-np.mean(W,axis=1)[:,None]
w_bar = w_bar.astype('float32')
#

plt.scatter(cordinates[0,:,0], cordinates[0,:,1])
plt.show() 
plt.close('all')
u, s_, v = np.linalg.svd(w_bar, full_matrices=False)
# print(s_)
s = np.diag(s_)[:3,:3]
u = u[:,0:3]
v = v[0:3,:]
#
# print(w_bar)

S_cap = np.dot(np.sqrt(s),v)
R_cap = np.dot(u,np.sqrt(s))

number_of_frame = nFrames
#
R_cap_i = R_cap[0:number_of_frame,:]
R_cap_j = R_cap[number_of_frame:2*number_of_frame, :]
#
# # # Calculating R from R_cap
A = np.zeros((2*number_of_frame,6))
#
for i in range (number_of_frame):
    A[2*i,0] = (R_cap_i[i,0]**2) - (R_cap_j[i,0]**2)
    A[2 * i, 1] = 2 * ((R_cap_i[i, 0] * R_cap_i[i, 1]) - (R_cap_j[i, 0] * R_cap_j[i, 1]))
    A[2*i,2] =  2 * ((R_cap_i[i, 0] * R_cap_i[i, 2]) - (R_cap_j[i, 0] * R_cap_j[i, 2]))
    A[2*i,3] =  (R_cap_i[i,1]**2) - (R_cap_j[i,1]**2)
    A[2*i,5] =  (R_cap_i[i,2]**2) - (R_cap_j[i,2]**2)
    A[2*i,4] = 2*((R_cap_i[i, 2] * R_cap_i[i, 1]) - (R_cap_j[i, 2] * R_cap_j[i, 1]))

    A[2*i+1,0] = R_cap_i[i,0]*R_cap_j[i,0]
    A[2*i+1,1] = R_cap_i[i,1]*R_cap_j[i,0] + R_cap_i[i,0]*R_cap_j[i,1]
    A[2*i+1,2] = R_cap_i[i,2]*R_cap_j[i,0] + R_cap_i[i,0]*R_cap_j[i,2]
    A[2 * i + 1, 3] = R_cap_i[i, 1] * R_cap_j[i, 1]
    A[2*i+1,4] = R_cap_i[i,2]*R_cap_j[i,1] + R_cap_i[i,1]*R_cap_j[i,2]
    A[2*i+1,5] = R_cap_i[i, 2] * R_cap_j[i, 2]
#
U , SIG , V = np.linalg.svd(A, full_matrices=False)
v = (V.T)[:,-1]
print(v, SIG)
QQT = np.zeros((3,3))

QQT[0,0] = v[0]
QQT[1,1] = v[3]
QQT[2,2] = v[5]

QQT[0,1] = v[1]
QQT[1,0] = v[1]


QQT[0,2] = v[2]
QQT[2,0] = v[2]

QQT[2,1] = v[4]
QQT[1,2] = v[4]

Q = np.linalg.cholesky(QQT)

R = np.dot(R_cap,Q)
print(np.dot(R[0,:],R[number_of_frame,:]))

Q_inv = np.linalg.inv(Q)

S = np.dot(Q_inv,S_cap)

X = S[0,:]
Y = S[1,:]
Z = S[2,:]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,Z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig.savefig('tracking.png', bbox_inches='tight')
plt.show()
#
#
