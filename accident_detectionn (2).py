#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


# Upgrade pip
get_ipython().system('pip install --upgrade pip')

# Install necessary libraries
get_ipython().system('pip install numpy pandas matplotlib tensorflow opencv-python')


# In[3]:


get_ipython().system('pip install numpy==1.22.0 pandas==1.3.5')


# In[4]:


get_ipython().system('pip install numpy==1.23.5')


# In[5]:


import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


# In[6]:


batch_size = 100
img_height = 250
img_width = 250


# In[7]:


training_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'C:\Users\cheer\OneDrive\Desktop\archive\data\train',
    seed=101,
    image_size= (img_height, img_width),
    batch_size=batch_size

)

testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'C:\Users\cheer\OneDrive\Desktop\archive\data\test',
    seed=101,
    image_size= (img_height, img_width),
    batch_size=batch_size)

validation_ds =  tf.keras.preprocessing.image_dataset_from_directory(
    r'C:\Users\cheer\OneDrive\Desktop\archive\data\val',
    seed=101,
    image_size= (img_height, img_width),
    batch_size=batch_size)


# In[8]:


class_names = training_ds.class_names

## Configuring dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
testing_ds = testing_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[9]:


img_shape = (img_height, img_width, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False


# In[10]:


model = tf.keras.Sequential([
    base_model,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Conv2D(128, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(len(class_names), activation= 'softmax')
])


# In[11]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[12]:


history = model.fit(training_ds, validation_data = validation_ds, epochs = 50)


# In[13]:


plt.plot(history.history['loss'], label = 'training loss')
plt.plot(history.history['accuracy'], label = 'training accuracy')
plt.grid(True)
plt.legend()


# In[14]:


plt.plot(history.history['val_loss'], label = 'validation loss')
plt.plot(history.history['val_accuracy'], label = 'validation accuracy')
plt.grid(True)
plt.legend()


# In[15]:


AccuracyVector = []
plt.figure(figsize=(40, 40))
for images, labels in testing_ds.take(1):
    predictions = model.predict(images)
    predlabel = []
    prdlbl = []

    for mem in predictions:
        predlabel.append(class_names[np.argmax(mem)])
        prdlbl.append(np.argmax(mem))

    AccuracyVector = np.array(prdlbl) == labels
    for i in range(40):
        ax = plt.subplot(10, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title('Pred: '+ predlabel[i]+' actl:'+class_names[labels[i]] )
        plt.axis('off')
        plt.grid(True)


# In[16]:


pip install opencv-python numpy tensorflow twilio


# In[17]:


pip install tkinter opencv-python numpy tensorflow twilio


# In[18]:


pip install twilio


# In[ ]:


import os
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from twilio.rest import Client
import sqlite3

# Initialize or connect to the database
def init_db():
    conn = sqlite3.connect('accident_detection.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS accident_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_path TEXT,
        result TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

# Insert result into the database
def log_result(video_path, result):
    conn = sqlite3.connect('accident_detection.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO accident_log (video_path, result) VALUES (?, ?)', (video_path, result))
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Load the trained deep learning model
MODEL_PATH = r"C:\Users\cheer\Downloads\Model.h5"  # Update with your model path
model = tf.keras.models.load_model(MODEL_PATH)

# Twilio Credentials (Replace with your actual credentials)
ACCOUNT_SID = "cccccccccc"
AUTH_TOKEN = "iiiiiiii"
TWILIO_NUMBER = "pppppp"
RECIPIENT_NUMBER = "+918919xxxxx"

# Function to preprocess a video frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (250, 250))  # Resize for model
    frame = frame / 255.0  # Normalize
    return frame

# Function to send SMS alert
def send_sms_alert():
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(
            body="🚨 Accident Detected! Immediate attention required.",
            from_=TWILIO_NUMBER,
            to=RECIPIENT_NUMBER
        )
        print(f"📩 SMS Alert Sent! Message SID: {message.sid}")
    except Exception as e:
        print(f"❌ Error sending SMS: {e}")

# Function to detect an accident in the video
def detect_accident(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    motion_threshold = 20.0  # Adjust as needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = preprocess_frame(frame)
        input_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension

        # Predict accident probability
        prediction = model.predict(input_frame)
        accident_prob = prediction[0][0]

        # Compute motion difference if previous frame exists
        motion_intensity = 0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, frame_gray)
            motion_intensity = np.mean(diff)

        prev_gray = frame_gray  # Update previous frame

        print(f"🔍 Accident Probability: {accident_prob:.4f}, Motion Intensity: {motion_intensity:.2f}")

        # Condition: Either model detects an accident OR high motion intensity
        if accident_prob > 0.5 or motion_intensity > motion_threshold:
            print("⚠️ Accident Detected!")
            accident_frame_path = "accident_detected_frame.jpg"
            cv2.imwrite(accident_frame_path, frame * 255)  # Save frame
            send_sms_alert()
            cap.release()
            return "Accident Detected 🚨", accident_frame_path

    cap.release()
    return "No Accident Detected ✅", None

# Function to browse and select a video file
def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")]
    )
    if file_path:
        video_path.set(file_path)

# Function to process the selected video
def process_video():
    if not video_path.get():
        messagebox.showwarning("Warning", "Please select a video file first.")
        return

    result, accident_frame_path = detect_accident(video_path.get())
    messagebox.showinfo("Detection Result", result)

    # Log result to the database
    log_result(video_path.get(), result)
    print(f"Logged to DB: {video_path.get()} - {result}")

    # If an accident is detected, display the accident frame
    if accident_frame_path:
        display_image(accident_frame_path)

# Function to display the detected accident frame
def display_image(image_path):
    image = Image.open(image_path)
    image = image.resize((300, 200), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)

    img_label.config(image=photo)
    img_label.image = photo
    img_label.pack(pady=10)

# Create Tkinter GUI
root = tk.Tk()
root.title("Accident Detection System")
root.geometry("500x500")
root.configure(bg="lightgray")

video_path = tk.StringVar()

# Title Label
title_label = tk.Label(root, text="🚗 Accident Detection System", font=("Arial", 16, "bold"), fg="red", bg="lightgray")
title_label.pack(pady=20)

# File Selection
browse_button = tk.Button(root, text="Browse Video", font=("Arial", 12), command=browse_file)
browse_button.pack(pady=10)

video_label = tk.Label(root, textvariable=video_path, font=("Arial", 10), wraplength=400, bg="lightgray")
video_label.pack()

# Process Button
process_button = tk.Button(root, text="Detect Accident", font=("Arial", 12, "bold"), bg="red", fg="white", command=process_video)
process_button.pack(pady=20)

# Image Label (for detected accident frame)
img_label = tk.Label(root, bg="lightgray")
img_label.pack()

# Run Tkinter loop
root.mainloop()


# In[20]:


import sqlite3

def view_logs():
    conn = sqlite3.connect('accident_detection.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM accident_log")
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        print(row)

view_logs()


# In[ ]:




