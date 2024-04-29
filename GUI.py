import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os
import glob

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection")
        self.root.geometry('1900x1060')  # Set the window size

        # Load YOLO model
        self.model = YOLO('best.pt')

        # Setup GUI layout
        self.frame = Frame(root)
        self.frame.pack(pady=10)

        # Load button
        self.load_button = Button(self.frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=10)

        # Detect button
        self.detect_button = Button(self.frame, text="Detect Image", command=self.detect_image, state=tk.DISABLED)
        self.detect_button.pack(side=tk.LEFT, padx=10)

        # Display panel for original image
        self.panel = Label(root)
        self.panel.place(x=20, y=80)  # Using 'place' for fixed positioning

        # Display panel for detected image
        self.panel2 = Label(root)
        self.panel2.place(x=640, y=80)  # Using 'place' for fixed positioning

        # Path to the current loaded image
        self.loaded_image_path = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        self.loaded_image_path = file_path  # Store the image path as an instance attribute

        if any(file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            img = Image.open(file_path)
            img2 = img.resize((580, 580), Image.Resampling.LANCZOS)  # Resize image
            img_tk = ImageTk.PhotoImage(img2)
            self.panel.config(image=img_tk)
            self.panel.image = img_tk  # Keep a reference
            self.detect_button.config(state=tk.NORMAL)  # Enable detect button

    def detect_image(self):
        if self.loaded_image_path:
            # Perform detection using the model
            img = cv2.imread(self.loaded_image_path)
            results = self.model(img, save=True)

            # Find the latest folder in the runs/detect directory
            latest_folder = max(glob.glob(os.path.join('runs/detect', '*/')), key=os.path.getctime)
            
            # Find the latest image file within that folder
            latest_image_path = max(glob.glob(os.path.join(latest_folder, '*.jpg')), key=os.path.getctime)
            
            # Load and display the latest image with detections
            img = Image.open(latest_image_path)
            img = img.resize((580, 580), Image.Resampling.LANCZOS)  # Resize image
            img_tk = ImageTk.PhotoImage(img)
            self.panel2.config(image=img_tk)
            self.panel2.image = img_tk  # Keep a reference
        else:
            print("No image loaded")  # Optionally handle the case where no image is loaded

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()
