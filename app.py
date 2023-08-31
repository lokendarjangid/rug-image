from flask import Flask, request, send_file, send_from_directory
import cv2
import os
import numpy as np
from io import BytesIO
import mediapipe as mp

app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Add these lines to import and initialize the face mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)


def calculate_face_tilt(face_landmarks, ih, iw):
    left_eye = np.array([face_landmarks.landmark[33].x *
                        iw, face_landmarks.landmark[33].y * ih])
    right_eye = np.array(
        [face_landmarks.landmark[263].x * iw, face_landmarks.landmark[263].y * ih])
    nose_tip = np.array([face_landmarks.landmark[1].x * iw,
                        face_landmarks.landmark[1].y * ih])

    # Calculate left-right tilt angle
    eye_angle = np.degrees(np.arctan2(
        right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # Calculate up-down tilt angle
    eye_center = (left_eye + right_eye) / 2
    nose_angle = np.degrees(np.arctan2(
        nose_tip[1] - eye_center[1], nose_tip[0] - eye_center[0]))

    return eye_angle, nose_angle


def overlay_product(person_img, product_img):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(
            cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = person_img.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin *
                                                       ih), int(bboxC.width * iw), int(bboxC.height * ih)

                top = y - int(h * 0.9)
                bottom = y + h - int((h/2) * 0.1)
                left = x - int(w * 0.17)
                right = x + w + int(w * 0.18)

                width = right - left
                height = bottom - top

                # Calculate face angles using face landmarks
                face_landmarks = face_mesh.process(cv2.cvtColor(
                    person_img, cv2.COLOR_BGR2RGB)).multi_face_landmarks[0]
                eye_angle, nose_angle = calculate_face_tilt(
                    face_landmarks, ih, iw)

                # Rotate the product image according to the face angles (negate the eye_angle)
                M = cv2.getRotationMatrix2D(
                    (width // 2, height // 2), -eye_angle, 1)
                product_resized = cv2.resize(product_img, (width, height))
                product_rotated = cv2.warpAffine(
                    product_resized, M, (width, height))

                # Adjust the product position based on the up-down tilt angle
                vertical_shift = int(
                    height * 0.1 * np.sin(np.radians(nose_angle)))
                top += vertical_shift
                bottom += vertical_shift

                if product_rotated.shape[2] == 4:
                    alpha = product_rotated[:, :, 3] / 255.0
                    alpha = np.expand_dims(alpha, axis=2)

                    if person_img[top:top + height, left:left + width].shape[:2] == product_rotated[:, :, :3].shape[:2]:
                        person_img[top:top + height, left:left + width] = (
                            1 - alpha) * person_img[top:top + height, left:left + width] + alpha * product_rotated[:, :, :3]
                else:
                    if person_img[top:top + height, left:left + width].shape == product_rotated.shape:
                        person_img[top:top + height,
                                   left:left + width] = product_rotated

    return person_img


def overlay_frame(result_img, frame_path):
    # Read the frame image with transparency (alpha channel)
    frame_img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

    # Resize the frame image to match the dimensions of the result image
    frame_resized = cv2.resize(
        frame_img, (result_img.shape[1], result_img.shape[0]))

    # Overlay the frame image on the result image, taking into account the alpha channel
    if frame_resized.shape[2] == 4:
        alpha = frame_resized[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)

        result_img = (1 - alpha) * result_img + alpha * frame_resized[:, :, :3]

    return result_img.astype(np.uint8)


@app.route('/overlay', methods=['POST'])
def overlay():
    person_img = request.files['person']

    person_img = cv2.imdecode(np.frombuffer(
        person_img.read(), np.uint8), cv2.IMREAD_COLOR)

    # Load the product image from the specified path
    product_img = cv2.imread('image/cap.png', cv2.IMREAD_UNCHANGED)

    result = overlay_product(person_img, product_img)

    # Call the overlay_frame function to add the frame image
    frame_path = 'image/frame.png'
    result_with_frame = overlay_frame(
        result, frame_path)

    buffer = BytesIO()
    _, encoded = cv2.imencode(
        '.jpg', result_with_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    buffer.write(encoded)

    buffer.seek(0)

    return send_file(buffer, mimetype='image/jpeg')


@app.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory('image', filename)


@app.route('/')
def index():
    return send_file('index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
