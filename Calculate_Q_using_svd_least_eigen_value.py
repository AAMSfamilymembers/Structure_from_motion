import numpy as np
import cv2
import matplotlib
matplotlib.get_backend()

number_of_frame = 20
sift = cv2.xfeatures2d.SIFT_create()
cap = cv2.VideoCapture("/home/ubuntu/PycharmProjects/Structure_from_motion/cv/current/456.mp4")

# params for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# find corners in the first frame
_ , frame = cap.read()
old_frame = frame
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

kp = sift.detect(old_gray,None)
p0 = (np.array(list(map(lambda p: [p.pt], kp))).astype(int)).astype(np.float32)[:5]
mask = np.zeros_like(old_frame)
color = np.random.randint(0,255,(p0.shape[0],3))
w = np.zeros((2*number_of_frame,len(p0)))
print(frame.shape)
j=0
cordinates = []
while(j!=number_of_frame):
    _, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(j)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    temp = np.squeeze(p1)
    print(temp.shape)
    ind = np.where(st==1)
    print(ind[0].shape)
    w[j,ind[0]] = ((temp.T)[0])[ind[0]]
    w[j + number_of_frame, ind[0]] = ((temp.T)[1])[ind[0]]
    print(st.shape)
    #print(np.squeeze(p1).shape)
    #print(np.squeeze(p0).shape)
    #print("\n")

    # draw the tracks
    for i,(new,old) in enumerate(zip(p1, p0)):
        a,b = new.ravel()
        c,d = old.ravel()
        if ((a-c)**2 + (b-d)**2)**0.5 > 1:
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

    img = cv2.add(frame,mask)
    cv2.imshow('tracks',mask)
    cv2.imshow("frame", frame)
    cordinates.append(p0)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = p1.reshape(-1,1,2)
    j=j+1

w_bar = w - np.mean(w,axis=0)
w_bar = w_bar.astype('float32')


u, s_, v = np.linalg.svd(w_bar, full_matrices=False)
s = np.diag(s_)[:3,:3]
u = u[:,0:3]
v = v[0:3,:]

S_cap = np.dot(np.sqrt(s),v[0:3,:])
R_cap = np.dot(u[:,0:3],np.sqrt(s))
print("norm")

norm = np.linalg.norm(R_cap, ord=None, axis=1, keepdims=True)

R_cap = R_cap / norm

print(R_cap.shape)

R_cap_i = R_cap[0:number_of_frame,:]
R_cap_j = R_cap[number_of_frame:2*number_of_frame, :]


# Calculating R from R_cap

zero = np.zeros((number_of_frame, 6))
A = np.zeros((number_of_frame,6))

for i in range(number_of_frame):
    for j in range(3):
        A[i,j] = (R_cap_i[i,j]**2) - (R_cap_j[i,j]**2) + (R_cap_i[i,j] * R_cap_j[i,j])
    A[i, 3] = 2 * ((R_cap_i[i, 0] * R_cap_i[i, 1]) - (R_cap_j[i, 0] * R_cap_j[i, 1])) + (R_cap_i[i, 0] * R_cap_j[i, 1]) + (R_cap_i[i, 1] * R_cap_j[i, 0])
    A[i, 4] = 2 * ((R_cap_i[i, 1] * R_cap_i[i, 2]) - (R_cap_j[i, 1] * R_cap_j[i, 2])) + (R_cap_i[i, 1] * R_cap_j[i, 2]) + (R_cap_i[i, 2] * R_cap_j[i, 1])
    A[i, 5] = 2 * ((R_cap_i[i, 2] * R_cap_i[i, 0]) - (R_cap_j[i, 2] * R_cap_j[i, 0])) + (R_cap_i[i, 2] * R_cap_j[i, 0]) + (R_cap_i[i, 0] * R_cap_j[i, 2])

U , SIG , V = np.linalg.svd(A, full_matrices=False)
v = (V[:,-1])
#print(v.shape)
QQT =np.zeros((3,3))
for i in range(3):
    QQT[i,i] = v[i]

QQT[0,1] = v[3]
QQT[1,0] = v[3]

QQT[0,2] = v[5]
QQT[2,0] = v[5]

QQT[2,1] = v[4]
QQT[1,2] = v[4]
'''
index = np.where(QQT<0)
QQT[index] = -1 * QQT[index]
'''
print("QQT")
print(QQT)
#Q = np.linalg.cholesky(QQT)

Q = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        if j<=i:
            sum = 0
            if j==i:
                for k in range(3):
                    if k<j or k==0:
                        sum = sum + (Q[j][k]**2)
                Q[j][j] = np.sqrt(QQT[j][j] - sum)
            else:
                for k in range(3):
                    sum = sum + (Q[i][k] * Q[j][k])
                Q[i][j] = (QQT[i][j] - sum) / Q[j][j]

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
