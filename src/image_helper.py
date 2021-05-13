'''
Util for image manipulation
'''
import cv2

DEFAULT_COLOR = (0, 255, 0)
DEFAULT_COLOR_GAZE = (255, 255, 255)
DEFAULT_THICKNESS = 2

EYE_SPAN = 2
CROP_EYE_SPAN = 30
GAZE_EXP_COEF = 100  # Gaze Expansion Coefficient


def get_shape(image):
    height, width, _ = image.shape
    return height, width


def get_face_coordinates(face_image, dims):
    height, width = dims
    _, _, score, x1, y1, x2, y2 = face_image

    x1 = int(width * x1)
    x2 = int(width * x2)
    y1 = int(height * y1)
    y2 = int(height * y2)
    return (x1, y1), (x2, y2), score


def get_eyes_coordinates(face, eyes):
    height, width = get_shape(face)
    x1, y1, x2, y2 = eyes

    x1 = int(width * x1)
    x2 = int(width * x2)
    y1 = int(height * y1)
    y2 = int(height * y2)
    return (x1, y1), (x2, y2)


def draw_face_box(image, face, threshold):
    height, width = get_shape(image)
    p1, p2, score = get_face_coordinates(face, (height, width))

    if score >= threshold:
        colour = DEFAULT_COLOR

    image = cv2.rectangle(image, p1, p2, colour, 2)
    return image


def draw_eyes_boxes(face, eyes):
    e1, e2 = get_eyes_coordinates(face, eyes)

    left_eye = ((e1[0]-EYE_SPAN, e1[1]-EYE_SPAN),
                (e1[0]+EYE_SPAN, e1[1]+EYE_SPAN))
    right_eye = ((e2[0]-EYE_SPAN, e2[1]-EYE_SPAN),
                 (e2[0]+EYE_SPAN, e2[1]+EYE_SPAN))
    output_face = cv2.rectangle(
        face, left_eye[0], left_eye[1], DEFAULT_COLOR, DEFAULT_THICKNESS)
    output_face = cv2.rectangle(
        output_face, right_eye[0], right_eye[1], DEFAULT_COLOR, DEFAULT_THICKNESS)
    return output_face


def crop_face(image, face):
    height, width = get_shape(image)
    (x1, y1), (x2, y2), _ = get_face_coordinates(face, (height, width))
    slice_x = slice(x1, x2)
    slice_y = slice(y1, y2)
    return image[slice_y, slice_x], (x1, y1)


def crop_eyes(image, upper_corner, eyes):
    el, er = eyes
    y, x = upper_corner

    eyel_x = (el[0]-CROP_EYE_SPAN+x, el[0]+CROP_EYE_SPAN+x)
    eyel_y = (el[1]-CROP_EYE_SPAN+y, el[1]+CROP_EYE_SPAN+y)
    eyer_x = (er[0]-CROP_EYE_SPAN+x, er[0]+CROP_EYE_SPAN+x)
    eyer_y = (er[1]-CROP_EYE_SPAN+y, er[1]+CROP_EYE_SPAN+y)

    el_silcex = slice(eyel_x[0], eyel_x[1])
    el_silcey = slice(eyel_y[0], eyel_y[1])
    er_silcex = slice(eyer_x[0], eyer_x[1])
    er_silcey = slice(eyer_y[0], eyer_y[1])

    eye_crop_left = image[el_silcex, el_silcey]
    eye_crop_right = image[er_silcex, er_silcey]
    return eye_crop_left, eye_crop_right


def plot_gaze(input_image, gaze, left_eye, right_eye):

    point_1 = int(left_eye[0] + GAZE_EXP_COEF * gaze[0]
                  ), int(left_eye[1] - GAZE_EXP_COEF * gaze[1])
    point_2 = int(right_eye[0] + GAZE_EXP_COEF * gaze[0]
                  ), int(right_eye[1] - GAZE_EXP_COEF * gaze[1])

    gaze_image = cv2.line(input_image, left_eye, point_1,
                          DEFAULT_COLOR_GAZE, DEFAULT_THICKNESS)
    gaze_image = cv2.line(gaze_image, right_eye, point_2,
                          DEFAULT_COLOR_GAZE, DEFAULT_THICKNESS)
    return gaze_image
