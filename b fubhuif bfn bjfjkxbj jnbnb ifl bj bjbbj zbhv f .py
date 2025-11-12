import cv2
import mediapipe as mp
import time
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

filter = [
    None, 
    'GRAYSCALE',
    'SEPIA',
    'NEGATIVE',
    'BLUR'
]
current_filter = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

    last_action_time = 0
    debounce_time = 1.0

    def apply_filter(frame, filter_type):
        if filter_type == 'GRAYSCALE':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif filter_type == 'SEPIA':
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            return cv2.transform(frame, kernel)
        elif filter_type == 'NEGATIVE':
            return cv2.bitwise_not(frame)
        elif filter_type == 'BLUR':
            return cv2.GaussianBlur(frame, (15, 15), 0)
        else:
            return frame
        
while True:
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            if thumb_tip.y < index_tip.y:
                current_time = time.time()
                if current_time - last_action_time > debounce_time:
                    current_filter = (current_filter + 1) % len(filter)
                    last_action_time = current_time

    frame = apply_filter(frame, filter[current_filter])
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()