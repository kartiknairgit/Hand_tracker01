import cv2
import mediapipe as mp
import subprocess
import tkinter as tk
from tkinter import StringVar
from PIL import Image, ImageTk
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

# Initialize variables to keep track of the gesture and time
last_finger_count = 0
gesture_start_time = None
gesture_duration = 3  # Time in seconds to hold the gesture

def perform_action(finger_count, action_var):
    if finger_count == 1:
        subprocess.run(['osascript', '-e', 'tell application "Music" to playpause'])
        action_var.set("One finger: Play/Pause")
    elif finger_count == 2:
        subprocess.run(['osascript', '-e', 'tell application "Music" to set sound volume to (sound volume + 10)'])
        action_var.set("Two fingers: Increase Volume")
    elif finger_count == 3:
        subprocess.run(['osascript', '-e', 'tell application "Music" to set sound volume to (sound volume - 10)'])
        action_var.set("Three fingers: Decrease Volume")
    elif finger_count == 4:
        subprocess.run(['osascript', '-e', 'tell application "Music" to next track'])
        action_var.set("Four fingers: Next Track")
    elif finger_count == 5:
        subprocess.run(['osascript', '-e', 'tell application "Music" to previous track'])
        action_var.set("Five fingers: Previous Track")
    else:
        action_var.set("No fingers detected")

root = tk.Tk()
root.title("Hand Gesture Recognition")
root.geometry("800x600")

action_var = StringVar()
action_var.set("No gesture detected yet")

action_label = tk.Label(root, textvariable=action_var, font=("Helvetica", 16))
action_label.pack(pady=20)

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

def update_gui():
    global last_finger_count, gesture_start_time
    
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        root.after(10, update_gui)
        return
    
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            finger_count = 0
            
            for tip_id in finger_tips:
                if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                    finger_count += 1
            
            if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 2].x:
                finger_count += 1

            # Check if the same gesture is held for a minimum duration
            if finger_count == last_finger_count:
                if gesture_start_time is None:
                    gesture_start_time = time.time()
                elif time.time() - gesture_start_time >= gesture_duration:
                    perform_action(finger_count, action_var)
                    gesture_start_time = None  
            else:
                last_finger_count = finger_count
                gesture_start_time = None 

    img = Image.fromarray(image_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    canvas.imgtk = imgtk  

    root.after(10, update_gui)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    update_gui()
    root.mainloop()

cap.release()
cv2.destroyAllWindows()