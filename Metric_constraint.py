import numpy as np
import cv2
import importlib
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
matplotlib.get_backend()


# Using sparse optical flow
number_of_frame = 4
# sift = cv2.xfeatures2d.SIFT_create()
# cap = cv2.VideoCapture("/home/abhay/PycharmProjects/structure_from_motion/venv/VID_20190713_220210.mp4")
#
# # params for lucas kanade optical flow
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#
#
# # find corners in the first frame
# _ , frame = cap.read()
# old_frame = frame
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#
# kp = sift.detect(old_gray,None)
# p0 = (np.array(list(map(lambda p: [p.pt], kp))).astype(int)).astype(np.float32)
# mask = np.zeros_like(old_frame)
# color = np.random.randint(0,255,(p0.shape[0],3))
# w = np.zeros((2*number_of_frame,len(p0)))
# j=0
# cordinates = []
# while(j!=number_of_frame):
#     _, frame = cap.read()
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#     temp = np.squeeze(p1)
#     w[j] = (temp.T)[0]
#     w[j + number_of_frame] = (temp.T)[1]
#     # print(w)
#     j=j+1
# Using manual points
w = np.asarray([[1360,1395,1389,1356],[1339,1385,1387,1342],[1299,1369,1363,1293],[1237,1325,1323,1237],[555,545,631,646],[556,547,636,646],[585,584,670,677],[563,560,639,651]])
w_bar = w - np.mean(w,axis=1)[:,None]
w_bar = w_bar.astype('float32')


u, s_, v = np.linalg.svd(w_bar, full_matrices=False)

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

U , SIG , V = np.linalg.svd(A, full_matrices=False)
v = (V[:,-1])
QQT = np.zeros((3,3))
print(v)

QQT[0,0] = v[0]
QQT[1,1] = v[3]
QQT[2,2] = v[5]

QQT[0,1] = v[1]
QQT[1,0] = v[1]


QQT[0,2] = v[2]
QQT[2,0] = v[2]

QQT[2,1] = v[4]
QQT[1,2] = v[4]

print(QQT)
Q = np.linalg.cholesky(QQT)

R = np.dot(R_cap,Q)
# print(np.dot(R[0,:],R[number_of_frame,:]))

Q_inv = np.linalg.inv(Q)

S = np.dot(Q_inv,S_cap)

# #print(p1[len(p1)-1,0,0])
#
# #print(p1[len(p1)-1,0,1])
# ax.surf(S[0,:], S[1,:],S[2,:])
X = S[0,:]
Y = S[1,:]
Z = S[2,:]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,Z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig.savefig('/home/abhay/Documents/optical_flow2.png', bbox_inches='tight')
# np.save('Shape.npy',S_cap)
# np.savetxt("Shape.csv", S_cap, delimiter=",")
# cv2.destroyAllWindows()'''