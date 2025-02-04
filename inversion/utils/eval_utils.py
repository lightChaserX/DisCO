import numpy as np

MN = {
    'silhouette': [
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ],

    'lipsUpperOuter': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    'lipsLowerOuter': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    'lipsUpperInner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    'lipsLowerInner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

    'rightEyeUpper0': [246, 161, 160, 159, 158, 157, 173],
    'rightEyeLower0': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    'rightEyeUpper1': [247, 30, 29, 27, 28, 56, 190],
    'rightEyeLower1': [130, 25, 110, 24, 23, 22, 26, 112, 243],
    'rightEyeUpper2': [113, 225, 224, 223, 222, 221, 189],
    'rightEyeLower2': [226, 31, 228, 229, 230, 231, 232, 233, 244],
    'rightEyeLower3': [143, 111, 117, 118, 119, 120, 121, 128, 245],

    'rightEyebrowUpper': [156, 70, 63, 105, 66, 107, 55, 193],
    'rightEyebrowLower': [35, 124, 46, 53, 52, 65],

    'rightEyeIris': [473, 474, 475, 476, 477],

    'leftEyeUpper0': [466, 388, 387, 386, 385, 384, 398],
    'leftEyeLower0': [263, 249, 390, 373, 374, 380, 381, 382, 362],
    'leftEyeUpper1': [467, 260, 259, 257, 258, 286, 414],
    'leftEyeLower1': [359, 255, 339, 254, 253, 252, 256, 341, 463],
    'leftEyeUpper2': [342, 445, 444, 443, 442, 441, 413],
    'leftEyeLower2': [446, 261, 448, 449, 450, 451, 452, 453, 464],
    'leftEyeLower3': [372, 340, 346, 347, 348, 349, 350, 357, 465],

    'leftEyebrowUpper': [383, 300, 293, 334, 296, 336, 285, 417],
    'leftEyebrowLower': [265, 353, 276, 283, 282, 295],

    'leftEyeIris': [468, 469, 470, 471, 472],

    'midwayBetweenEyes': [168],

    'noseTip': [1],
    'noseBottom': [2],
    'noseRightCorner': [98],
    'noseLeftCorner': [327],

    'rightCheek': [205],
    'leftCheek': [425]
};


def extract3d_landmark(image, face_mesh):
    pts3D = np.zeros((468, 3))
    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        return None, None
        # import pdb; pdb.set_trace()
    face_landmarks = results.multi_face_landmarks[0]
    for i in range(468):
        pts3D[i, 0] = face_landmarks.landmark[i].x
        pts3D[i, 1] = face_landmarks.landmark[i].y
        pts3D[i, 2] = face_landmarks.landmark[i].z

    h, w = image.shape[0], image.shape[1]
    pts3D_rescale = pts3D[:,:2].copy()
    pts3D_rescale[:, 0] = pts3D[:, 0] * w
    pts3D_rescale[:, 1] = pts3D[:, 1] * h

    return pts3D, pts3D_rescale

def get_eyes(lm):
    lm_eye_right = np.concatenate((lm[MN['leftEyeLower1'], :], lm[MN['leftEyeUpper1'], :]), axis=0)  # left-clockwise
    lm_eye_left = np.concatenate((lm[MN['rightEyeLower1'], :], lm[MN['rightEyeUpper1'], :]), axis=0)  # left-clockwise
    lm_mouth_outer = np.concatenate((lm[MN['lipsLowerOuter'], :], lm[MN['lipsUpperOuter'], :]), axis=0)  # left-clockwise
    return lm_eye_right, lm_eye_left, lm_mouth_outer