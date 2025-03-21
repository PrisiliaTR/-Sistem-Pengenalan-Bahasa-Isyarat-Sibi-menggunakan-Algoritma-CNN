from src.backbone import TFLiteModel, get_model
from src.landmarks_extraction import mediapipe_detection, draw, extract_coordinates, load_json_file
from src.config import SEQ_LEN, THRESH_HOLD
Import Numpy as np
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

sign_to_prediction_map = {k.lower(): v for k, v in load_json_file("src/label_kelas_prediksi.json").items()}
prediction_to_sign_map = {v: k for k, v in load_json_file("src/label_kelas_prediksi.json").items()}

encode_sign = lambda x: sign_to_prediction_map.get(x.lower())
decode_prediction = lambda x: prediction_to_sign_map.get(x)

model_paths = ['./models/islr-fp16-192-8-seed_all42-foldall-last.h5']
isolated_sign_models = [get_model() for _ in model_paths]

# Load model weights
for model, path in zip(isolated_sign_models, model_paths):
    model.load_weights(path)

def real_time_sibi():
    """
    Run real-time sign recognition using the webcam.
    """
    recognized_signs = []
    tflite_model = TFLiteModel(islr_models=isolated_sign_models)
    sequence_buffer = []
    capture = cv2.VideoCapture(0)

    capture.set(cv2.Cap_Prop_Frame_Width, 1280)
    capture.set(cv2.Cap_Prop_Frame_Height, 720)

    resolution_width = int(capture.get(cv2.Cap_Prop_Frame_Width))
    resolution_height = int(capture.get(cv2.Cap_Prop_Frame_Height))
    print(f"Resolution: {resolution_width}x{resolution_height}")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        last_detected_sign = ""
        last_prediction_confidence = 0.0

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            processed_frame, detection_results = mediapipe_detection(frame, holistic)
            draw(processed_frame, detection_results)

            # Extract landmarks
            try:
                landmarks = extract_coordinates(detection_results)
            except:
                landmarks = np.zeros((468 + 21 + 33 + 21, 3))

            sequence_buffer.append(landmarks)

            # Perform prediction when the buffer reaches SEQ_LEN
            if len(sequence_buffer) % SEQ_LEN == 0:
                prediction = tflite_model(np.array(sequence_buffer, dtype=np.float32))["outputs"]
                if np.max(prediction.numpy(), axis=-1) > THRESH_HOLD:
                    last_prediction_confidence = np.max(prediction.numpy(), axis=-1)
                    predicted_sign = np.argmax(prediction.numpy(), axis=-1)
                    last_detected_sign = decode_prediction(predicted_sign)
                sequence_buffer = []

            # Display flipped frame
            processed_frame = cv2.flip(processed_frame, 1)

            cv2.putText(processed_frame, f"{len(sequence_buffer)}", (3, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if last_detected_sign and last_detected_sign not in recognized_signs:
                recognized_signs.insert(0, last_detected_sign)

            
            frame_height, frame_width = processed_frame.shape[0], processed_frame.shape[1]
            white_bar = np.ones((frame_height // 8, frame_width, 3), dtype='uint8') * 255
            processed_frame = np.concatenate((white_bar, processed_frame), axis=0)
            
            cv2.putText(processed_frame, f"{', '.join(str(sign) for sign in recognized_signs)}", (3, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(processed_frame, f"{last_prediction_confidence * 100:.2f}%", (500, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.putText(processed_frame, "Press Q to exit", (10, resolution_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(processed_frame, "Press C to clear", (10, resolution_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Sign Recognition', processed_frame)
            
            key = cv2.waitKey(10)
            if key & 0xFF == ord("q"):
                break
            elif key & 0xFF == ord("c"):
                recognized_signs.clear()
                last_prediction_confidence = 0.0

        capture.release()
        cv2.destroyAllWindows()

def main():
    real_time_sibi()

if __name__ == "__main__":
    main()
