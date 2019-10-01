from imutils import face_utils
import dlib
import cv2
import numpy as np
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
j = 0
ptsx = np.zeros((1, 51))
ptsy = np.zeros((1, 51))
wx = []
wy = []
while True:
    # Getting out image by webcam
    _, image = cap.read()
    # image = cv2.imread(str(j) + '.jpg')
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Get faces into webcam's image
    rects = detector(gray, 0)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # print(shape.size)
        # print(shape)
        j = j + 1
        # Draw on our image, all the finded cordinate points (x,y)
        c = 0
        for (x, y) in shape[17:]:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            ptsx[0, c] = x
            ptsy[0, c] = y
            c = c + 1
        wx.append(ptsx)
        wy.append(ptsy)
    # Show the image
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

# print((shape[int(shape.size/2)-1]))
# print((shape[int(shape.size/2)-1])[0])
W = np.zeros((2*j,51))
print((wx[1])[0,50])
for index in range(j):
    W[index,:] = (wx[index])[0,:]
    W[(j + index),:] = (wy[index])[0,:]

nFrames = j
nFeatures = 51
# for i in range(int(shape.size/2)):
#     cordinates.append(shape[i])
#
# # print(cordinates[0,:])
# nFrames, nFeatures = cordinates.shape[:2]
# cordinates = cordinates.reshape(nFrames, nFeatures, 2)
# # nFrames = 20
# print(cordinates.shape)
# W = np.zeros((2*nFrames, nFeatures))
#
# W[0:nFrames, :] = cordinates[:nFrames,:,0]
# W[nFrames:2*nFrames, :] = cordinates[:nFrames,:,1]
#
w_bar = W - np.mean(W, axis=1)[:, None]
w_bar = w_bar.astype('float32')
#

# plt.scatter(Wx[0].T, Wy[0].T)
# plt.savefig('books_read.png', bbox_inches='tight')
# plt.show()
# plt.close('all')
u, s_, v = np.linalg.svd(w_bar, full_matrices=False)
# print(s_)
s = np.diag(s_)[:3, :3]
u = u[:, 0:3]
v = v[0:3, :]
#
# print(w_bar)

S_cap = np.dot(np.sqrt(s), v)
R_cap = np.dot(u, np.sqrt(s))

number_of_frame = nFrames
#
R_cap_i = R_cap[0:number_of_frame, :]
R_cap_j = R_cap[number_of_frame:2 * number_of_frame, :]
#
# # # Calculating R from R_cap
A = np.zeros((2 * number_of_frame, 6))
#
i = 0
print(R_cap_i.shape)
for i in range(number_of_frame):
    A[2 * i, 0] = (R_cap_i[i, 0] ** 2) - (R_cap_j[i, 0] ** 2)
    A[2 * i, 1] = 2 * ((R_cap_i[i, 0] * R_cap_i[i, 1]) - (R_cap_j[i, 0] * R_cap_j[i, 1]))
    A[2 * i, 2] = 2 * ((R_cap_i[i, 0] * R_cap_i[i, 2]) - (R_cap_j[i, 0] * R_cap_j[i, 2]))
    A[2 * i, 3] = (R_cap_i[i, 1] ** 2) - (R_cap_j[i, 1] ** 2)
    A[2 * i, 5] = (R_cap_i[i, 2] ** 2) - (R_cap_j[i, 2] ** 2)
    A[2 * i, 4] = 2 * ((R_cap_i[i, 2] * R_cap_i[i, 1]) - (R_cap_j[i, 2] * R_cap_j[i, 1]))

    A[2 * i + 1, 0] = R_cap_i[i, 0] * R_cap_j[i, 0]
    A[2 * i + 1, 1] = R_cap_i[i, 1] * R_cap_j[i, 0] + R_cap_i[i, 0] * R_cap_j[i, 1]
    A[2 * i + 1, 2] = R_cap_i[i, 2] * R_cap_j[i, 0] + R_cap_i[i, 0] * R_cap_j[i, 2]
    A[2 * i + 1, 3] = R_cap_i[i, 1] * R_cap_j[i, 1]
    A[2 * i + 1, 4] = R_cap_i[i, 2] * R_cap_j[i, 1] + R_cap_i[i, 1] * R_cap_j[i, 2]
    A[2 * i + 1, 5] = R_cap_i[i, 2] * R_cap_j[i, 2]
#
U, SIG, V = np.linalg.svd(A, full_matrices=False)
v = (V.T)[:, -1]
print(v, SIG)
QQT = np.zeros((3, 3))

QQT[0, 0] = v[0]
QQT[1, 1] = v[3]
QQT[2, 2] = v[5]

QQT[0, 1] = v[1]
QQT[1, 0] = v[1]

QQT[0, 2] = v[2]
QQT[2, 0] = v[2]

QQT[2, 1] = v[4]
QQT[1, 2] = v[4]

Q = np.linalg.cholesky(QQT)

R = np.dot(R_cap, Q)
print("yay")
print(np.dot(R[0, :], R[number_of_frame, :]))

Q_inv = np.linalg.inv(Q)

S = np.dot(Q_inv, S_cap)

X = S[0, :]
Y = S[1, :]
Z = S[2, :]

ax = plt.gca(projection="3d")
ax.scatter(X,Y,Z, c='r',s=100)
ax.plot(X,Y,Z, color='r')

plt.show()

