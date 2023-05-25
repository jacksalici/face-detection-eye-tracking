import numpy as np
import math
import cv2


def distance(point_x, point_y):
    return math.sqrt(
        ((point_x[0] - point_y[0]) ** 2) + ((point_x[1] - point_y[1]) ** 2)
    )


def perpendicular(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def clamp(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def intersectionWithPlan(linePoint, lineDir, planOrth, planPoint):
    d = np.dot(np.subtract(linePoint, planPoint), planOrth) / (
        np.dot(lineDir, planOrth)
    )
    intersectionPoint = np.subtract(np.multiply(d, lineDir), linePoint)
    return intersectionPoint


class PnPSolver:
    def __init__(self) -> None:
        self.cameraMatrix = np.eye(3)
        self.distCoeffs = np.zeros((5, 1))
        self.eyeConst = 1.5

        self.facePOI3d = np.array(
            [
                [-6, 0, -8],  # right ear
                [6, 0, -8],  # left ear
                [0, -4, 2.5],  # nose
                [-5, -8, 0],  # right corner mouth
                [5, -8, 0],  # left corner mouth
                [-3, 0, 1],  # right eye
                [3, 0, 1],  # left eye
            ]
        )

    def _rotMatFromEye(self, eyeData):
        # print eyeData
        eyeDiameter = self.eyeConst * distance(eyeData[0], eyeData[1])
        eyeCenter = (
            (eyeData[1][0][0] + eyeData[1][1][0]) / 2.0,
            (eyeData[1][0][1] + eyeData[1][1][1]) / 2.0,
        )
        eyePos = eyeData[0]
        # HERE WE CONSTRUCT A MATRIX OF A BASE WHERE THE UNIT IS THE DIAMETER OF THE EYE AND AXIS OF THIS
        mainEyeAxis = (
            (eyeData[1][0][0] - eyeData[1][1][0]),
            (eyeData[1][0][1] - eyeData[1][1][1]),
        )
        secondEyeAxis = perpendicular(mainEyeAxis)

        reverseTransitionMatrix = (mainEyeAxis, secondEyeAxis)

        transitionMatrix = np.linalg.inv(reverseTransitionMatrix)
        eyeCenterInEyeRef = np.dot(transitionMatrix, eyeCenter)
        eyeCenterInEyeRef[1] = eyeCenterInEyeRef[1] + 0.2

        eyePosInEyeRef = np.dot(transitionMatrix, eyePos)

        eyeOffset = eyePosInEyeRef - eyeCenterInEyeRef

        eyeOffset = [clamp(eyeOffset[0], -0.99, 0.99), clamp(eyeOffset[1], -0.99, 0.99)]
        # Now we get the rotation values
        thetay = -np.arcsin(eyeOffset[0]) * self.eyeConst
        thetax = np.arcsin(eyeOffset[1]) * self.eyeConst
        # Aaand the rotation matrix
        rot = eulerAnglesToRotationMatrix([thetax, thetay, 0])
        # print rot
        return rot

    # Given the data from a faceExtract
    def getCoordFromFace(self, FacePOI, eyeData):
        # SOLVER FOR PNPs
        distCoeffs = np.zeros((5, 1))
        # HARDCODED CAM PARAMS
        width = 2402
        height = 1201
        maxSize = max(width, height)
        cameraMatrix = np.array(
            [[maxSize, 0, width / 2.0], [0, maxSize, height / 2.0], [0, 0, 1]], np.int32
        )
        

        
        retval, rvec, tvec = cv2.solvePnP(
            self.facePOI3d, FacePOI, cameraMatrix, distCoeffs
        )
        rt, jacobian = cv2.Rodrigues(rvec)
        rot2 = self._rotMatFromEye(eyeData)

        origin = [tvec[0][0], tvec[1][0], tvec[2][0]]
        headDir = np.dot(rot2, np.dot(rt, [0, 0, 1]))
        camPlaneOrthVector = [0, 0, 1]
        pointOnPlan = [0, 0, 0]
        return intersectionWithPlan(origin, headDir, camPlaneOrthVector, pointOnPlan)
