# digit_gui.py
# Draw a digit and predict using the trained CNN MNIST model

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# -------------------- 1️⃣ Define CNN model & load weights --------------------
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Load the trained CNN weights (from digit_recognizer.py)
model.load_weights("digit_model_weights.weights.h5")


# -------------------- 2️⃣ Tkinter GUI --------------------
class DrawPredictApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw a Digit (0-9)")
        self.canvas_width = 200
        self.canvas_height = 200

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        self.predict_button = tk.Button(master, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.label = tk.Label(master, text="Draw a digit and click Predict", font=("Helvetica", 14))
        self.label.pack()

        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 'white')
        self.draw_obj = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8  # brush size
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        self.draw_obj.ellipse([x-r, y-r, x+r, y+r], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_obj.rectangle([0,0,self.canvas_width,self.canvas_height], fill='white')
        self.label.config(text="Draw a digit and click Predict")

    def predict_digit(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)  # invert colors
        img_array = np.array(img)/255.0
        img_array = img_array.reshape(1,28,28,1)  # extra dimension for CNN
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        self.label.config(text=f"Predicted Digit: {predicted_digit}")


# -------------------- 3️⃣ Run the app --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawPredictApp(root)
    root.mainloop()
