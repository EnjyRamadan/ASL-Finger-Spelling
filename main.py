import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageTk
import numpy as np


class ModelEvaluatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Alphabet Predictor")
        self.root.geometry("600x600")
        self.root.configure(bg="#f4f4f9")

        # Frames for organization
        self.top_frame = tk.Frame(root, bg="#f4f4f9")
        self.top_frame.pack(pady=10)

        self.image_frame = tk.Frame(root, bg="#f4f4f9")
        self.image_frame.pack(pady=10)

        self.button_frame = tk.Frame(root, bg="#f4f4f9")
        self.button_frame.pack(pady=10)

        self.result_frame = tk.Frame(root, bg="#f4f4f9")
        self.result_frame.pack(pady=10)

        # Photo Label
        self.photo_label = tk.Label(self.top_frame, text="No photo selected.", font=("Arial", 12), bg="#f4f4f9")
        self.photo_label.pack()

        # Photo Canvas
        self.photo_canvas = tk.Canvas(self.image_frame, width=150, height=150, bg="#dfe6e9", relief=tk.RIDGE, bd=2)
        self.photo_canvas.pack()

        # Buttons
        self.select_photo_btn = tk.Button(
            self.button_frame, text="Select Photo", command=self.select_photo,
            bg="#0984e3", fg="white", font=("Arial", 12), width=15
        )
        self.select_photo_btn.grid(row=0, column=0, padx=5, pady=5)

        self.evaluate_btn = tk.Button(
            self.button_frame, text="Evaluate", command=self.evaluate_models, state=tk.DISABLED,
            bg="#ff6f91", fg="white", font=("Arial", 12), width=15
        )
        self.evaluate_btn.grid(row=0, column=1, padx=5, pady=5)

        # Treeview Table for Results
        self.result_table = ttk.Treeview(
            self.result_frame, columns=("Model Name", "Predicted Letter", "Confidence"), show="headings", height=10
        )
        self.result_table.heading("Model Name", text="Model Name")
        self.result_table.heading("Predicted Letter", text="Predicted Letter")
        self.result_table.heading("Confidence", text="Confidence")
        self.result_table.column("Model Name", width=200, anchor="center")
        self.result_table.column("Predicted Letter", width=150, anchor="center")
        self.result_table.column("Confidence", width=150, anchor="center")
        self.result_table.pack(pady=10)

        # Classification Label
        self.classification_label = tk.Label(
            self.result_frame, text="", font=("Arial", 14), bg="#f4f4f9", fg="#2d3436"
        )
        self.classification_label.pack(pady=10)

        # Internal variables
        self.photo_path = None
        self.displayed_image = None

        # Statically defined model paths
        self.model_paths = [
            "dense_121.h5",  # Replace with actual model paths
            "LSTM_2.h5",
            "CNN.h5",
            "Resnet.h5",
           
        ]

        # Class label mapping for A-Z
        self.class_labels = {i: chr(65 + i) for i in range(26)}  # Maps 0-25 to 'A'-'Z'

    def select_photo(self):
        self.photo_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if self.photo_path:
            self.photo_label.config(text=f"Selected Photo: {self.photo_path.split('/')[-1]}")
            self.display_photo()
            self.evaluate_btn.config(state=tk.NORMAL)

    def display_photo(self):
        img = Image.open(self.photo_path)
        img = img.resize((150, 150))  # Resize the image to fit the canvas
        self.displayed_image = ImageTk.PhotoImage(img)
        self.photo_canvas.create_image(0, 0, anchor=tk.NW, image=self.displayed_image)

    def evaluate_models(self):
        if not self.photo_path:
            messagebox.showwarning("Missing Input", "Please select a photo before evaluating.")
            return

        results = []

        try:
            # Preprocess the image
            img = load_img(self.photo_path, target_size=(64, 64))  # Adjust target_size as per your model
            img_array = img_to_array(img) / 255.0
            img_array = img_array.reshape((1, *img_array.shape))

            for model_path in self.model_paths:
                model_name = model_path.split('/')[-1]
                try:
                    model = load_model(model_path)
                    predictions = model.predict(img_array)
                    predicted_class_idx = np.argmax(predictions[0])  # Get the index of the highest confidence
                    confidence = predictions[0][predicted_class_idx]
                    predicted_class_label = self.class_labels.get(predicted_class_idx, "Unknown")
                    results.append((model_name, predicted_class_label, confidence))
                except Exception as e:
                    results.append((model_name, "Error", f"Error: {e}"))

            # Sort results by confidence
            results.sort(key=lambda x: x[2] if isinstance(x[2], float) else -1, reverse=True)

            # Display results in the Treeview table
            for row in self.result_table.get_children():
                self.result_table.delete(row)  # Clear any previous entries

            for model_name, predicted_letter, confidence in results:
                self.result_table.insert("", "end", values=(model_name, predicted_letter, f"{confidence:.4f}"))

            # Show the top prediction below the photo
            top_prediction = results[0] if results else None
            if top_prediction:
                self.classification_label.config(
                    text=f"Predicted Letter: {top_prediction[1]} (Confidence: {top_prediction[2]:.4f})",
                    fg="green" if top_prediction[2] > 0.8 else "red",
                )
            else:
                self.classification_label.config(text="No prediction available.", fg="red")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelEvaluatorApp(root)
    root.mainloop()
