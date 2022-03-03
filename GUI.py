from tkinter import *
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils
model = load_model('mp_hand_gesture')


f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()



root=Tk()

root.geometry("900x520")
root.configure(bg='black')
app=tk.Frame(root,height=500,width=850,bg='black',bd=5)
app.pack()


app.grid_rowconfigure(0, minsize = 600)
app.grid_columnconfigure(0, minsize = 1000)
image1 = Image.open(r'C:\Users\DIPRAJYOTI MAJUMDAR\Desktop\Hand_action_recognition\interface_files\istockphoto-1213063656-612x612.jpg')

image1 = image1.resize((850,500),Image.ANTIALIAS)

test = ImageTk.PhotoImage(image1)
label1=tk.Label(app,image=test)
label1.image=test
label1.place(x=0,y=0)
# tk.Button(app, text="Quit", command=root.destroy).grid(column=1, row=0)
label2= tk.Label(app,text='Show your hand gesture, AI will recognize and display it on screen',font=("times new roman",20,'bold'),fg='black')
label2.place(x=0,y=10)
l1=tk.Label(app,bg='red',borderwidth=5,relief='raised')
l1.place(x=10,y=70)



label= tk.Label(app,text='Hand CAM',font=("times new roman",15,'bold'),fg='black')
label.place(x=100,y=400)
root.maxsize(900,520)

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,200)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,350)

while True :
    try:
        _, frame = cap.read()
        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)

        img = ImageTk.PhotoImage(Image.fromarray(frame))
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        className = ''
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])



            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]



        l1['image']=img
        label['text']=className
        root.title("Hand Action Recognizer")
        icon = Image.open(r'C:\Users\DIPRAJYOTI MAJUMDAR\Desktop\Hand_action_recognition\interface_files\images.jfif')
        icon = ImageTk.PhotoImage(icon)
        root.iconphoto(False,icon)
        root.update()


    except:
        print('code=001001')
        break

cap.release()