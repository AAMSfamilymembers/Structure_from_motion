import numpy as np
import cv2
import importlib
import matplotlib
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
matplotlib.get_backend()

number_of_frame = 50
sift = cv2.xfeatures2d.SIFT_create()
cap = cv2.VideoCapture("/home/ubuntu/PycharmProjects/Structure_from_motion/cv/current/456.mp4")

#dataset = np.load('dataset_BGR1.npy')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 30,
                       qualityLevel = 0.8,
                       minDistance = 7,
                       blockSize = 7 )

# params for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# find corners in the first frame
_ , frame = cap.read()
old_frame = frame
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

kp = sift.detect(old_gray,None)
p0 = (np.array(list(map(lambda p: [p.pt], kp))).astype(int)).astype(np.float32)
mask = np.zeros_like(old_frame)
color = np.random.randint(0,255,(p0.shape[0],3))
w = np.zeros((2*number_of_frame,100))

j=1
while(j!=number_of_frame):
    _, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    print(st.shape)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        if ((a-c)**2 + (b-d)**2)**0.5 > 1:
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

    img = cv2.add(frame,mask)
    cv2.imshow('tracks',mask)
    cv2.imshow("frame", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    j=j+1
w_bar = w - np.mean(w,axis=0)
w_bar = w_bar.astype('float32')


u, s_, v = np.linalg.svd(w_bar, full_matrices=False)
s = np.diag(s_)[:3,:3]
u = u[:,0:3]
v = v[0:3,:]
print(s)

S_cap = np.dot(np.sqrt(s),v[0:3,:])
R_cap = np.dot(u[:,0:3],np.sqrt(s))
R_cap_i = R_cap[0:number_of_frame,:]
print(R_cap.shape)
R_cap_j = R_cap[number_of_frame:2*number_of_frame, :]

print(R_cap_i.shape,R_cap_j.shape)

print("w:\n")
print(R_cap_j.shape)
# Calculating R from R_cap

zero = np.zeros((number_of_frame, 6))
A = np.zeros((number_of_frame,6))

for i in range(number_of_frame):
    for j in range(3):
        A[i,j] = (R_cap_i[i,j]**2) - (R_cap_j[i,j]**2)
    A[i, 3] = 2 * (R_cap_i[i, 0] * R_cap_i[i, 1] - R_cap_j[i, 0] * R_cap_j[i, 1])
    A[i, 4] = 2 * (R_cap_i[i, 1] * R_cap_i[i, 2] - R_cap_j[i, 1] * R_cap_j[i, 2])
    A[i, 5] = 2 * (R_cap_i[i, 2] * R_cap_i[i, 0] - R_cap_j[i, 2] * R_cap_j[i, 0])

U , SIG , V = np.linalg.svd(A, full_matrices=False)
v = ((V.T)[:,-1])
print(v.shape)
QQT =np.zeros((3,3))
for i in range(3):
    QQT[i,i] = v[i]

QQT[0,1] = v[3]
QQT[1,0] = v[3]

QQT[0,2] = v[5]
QQT[2,0] = v[5]

QQT[2,1] = v[4]
QQT[1,2] = v[4]
print("QQT")
print(QQT)
Q = np.linalg.cholesky(QQT)
R = np.dot(R_cap,Q)
print(np.dot(R[0,:],R[number_of_frame,:]))

#print(p1[len(p1)-1,0,0])

#print(p1[len(p1)-1,0,1])
'''#ax.surf(S[0,:], S[1,:],S[2,:])
X = S_cap[0,:]
Y = S_cap[1,:]
Z = S_cap[2,:]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X,Y,Z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig.savefig('optical_flow2.png', bbox_inches='tight')
np.save('Shape.npy',S_cap)
np.savetxt("Shape.csv", S_cap, delimiter=",")
cv2.destroyAllWindows()'''
