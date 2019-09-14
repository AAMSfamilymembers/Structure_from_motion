import cv2
import numpy as np
import importlib
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
matplotlib.get_backend()

number_of_frame = 4
img1 = cv2.imread('/home/abhay/Documents/checker.jpeg')
img2 = cv2.imread('/home/abhay/Documents/checker1.jpg')
img3 = cv2.imread('/home/abhay/Documents/checker2.jpg')
img4 = cv2.imread('/home/abhay/Documents/checker3.jpg')

ret1, corners1 = cv2.findChessboardCorners(img1, (9,7))
print(corners1)
cv2.drawChessboardCorners(img1, (9,7), corners1, ret1)
cv2.namedWindow("", cv2.WINDOW_NORMAL)
cv2.imshow("", img1)
cv2.waitKey(0)
ret2, corners2 = cv2.findChessboardCorners(img2, (9,7))
cv2.drawChessboardCorners(img2, (9,7), corners2, ret2)
cv2.imshow("", img2)
cv2.waitKey(1)
ret3, corners3 = cv2.findChessboardCorners(img3, (9,7))
cv2.drawChessboardCorners(img3, (9,7), corners3, ret3)
cv2.imshow("", img3)
cv2.waitKey(1)
ret4, corners4 = cv2.findChessboardCorners(img4, (9,7))
cv2.drawChessboardCorners(img4, (9,7), corners4, ret4)
cv2.imshow("", img4)
cv2.waitKey(1)

corners1 = corners1.reshape(-1,2)
# print(corners1)
corners2 = corners2.reshape(-1,2)
corners3 = corners3.reshape(-1,2)
corners4 = corners4.reshape(-1,2)
#
#
w = np.asarray([corners1[:,0],corners2[:,0],corners3[:,0],corners4[:,0],corners1[:,1],corners2[:,1],corners3[:,1],corners4[:,1]])
#
w_bar = w - np.mean(w,axis=1)[:,None]
w_bar = w_bar.astype('float32')
#
#
u, s_, v = np.linalg.svd(w_bar, full_matrices=False)
#
s = np.diag(s_)[:3,:3]
u = u[:,0:3]
v = v[0:3,:]

S_cap = np.dot(np.sqrt(s),v)
R_cap = np.dot(u,np.sqrt(s))

R_cap_i = R_cap[0:number_of_frame,:]
R_cap_j = R_cap[number_of_frame:2*number_of_frame, :]

# # Calculating R from R_cap
A = np.zeros((2*number_of_frame,6))

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
fig.savefig('/home/abhay/Documents/optical_flow4.png', bbox_inches='tight')

