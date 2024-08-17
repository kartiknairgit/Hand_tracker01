import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

with mp_hands.Hands(
    max_num_hands=2,  
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.flip(image, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                finger_count = 0
                
                for tip_id in finger_tips:
                    if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                        finger_count += 1
                
                if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 2].x:
                    finger_count += 1

                if finger_count == 1:
                    print("One finger")
                elif finger_count == 2:
                    print("Two fingers")
                elif finger_count == 3:
                    print("Three fingers")
                elif finger_count == 4:
                    print("Four fingers")
                elif finger_count == 5:
                    print("Whole hand in frame")
                else:
                    print("No fingers detected")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit() 

        cv2.imshow('Hand Detection', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
