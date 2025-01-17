import cv2
import torch
from torchvision import transforms
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Label, Button, Frame
from src.train_model import CNNModel

# Load the pre-trained model
model = CNNModel()
model.load_state_dict(torch.load("./models/final_model.pth", weights_only=True))
model.eval()  # Set the model to evaluation mode

# Define the transformation for preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    tensor_image = transform(image).unsqueeze(0)  # Add batch dimension
    processed_image = transform(image)  # Return the processed image for display
    return tensor_image, processed_image

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor_image = transform(thresh).unsqueeze(0)
    return tensor_image

# Function to update the bar chart
def update_bar_chart(ax, canvas, probabilities):
    ax.clear()
    ax.barh(np.arange(10), probabilities, color="blue")
    ax.set_aspect(0.2)
    ax.set_yticks(np.arange(10))
    ax.set_yticklabels(np.arange(10))
    ax.set_title('Class Probability')
    ax.set_xlim(0, 1.1)
    canvas.draw()

def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    # Preprocess the image
    image_tensor, processed_image = preprocess_image(file_path)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        predicted_digit = output.argmax(dim=1).item()
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().numpy()

    # Display the predicted digit
    result_label.config(text=f"Predicted Digit: {predicted_digit}")

    # Display the preprocessed image scaled to 200x200 pixels
    processed_image_np = processed_image.numpy().squeeze() * 255
    processed_image_np = processed_image_np.astype(np.uint8)
    processed_img = ImageTk.PhotoImage(image=Image.fromarray(processed_image_np).resize((200, 200)))
    preprocessed_image_label.config(image=processed_img)
    preprocessed_image_label.image = processed_img

    # Update the probability bar chart
    ax.set_aspect(1.0)  # Match the aspect ratio to the image
    canvas.get_tk_widget().config(width=200, height=200)
    update_bar_chart(ax, canvas, probabilities)

def start_camera_feed():
    global running
    running = True
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    def update_frame():
        if not running:
            cap.release()
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            cap.release()
            return

        # Define a detection box
        height, width, _ = frame.shape
        bbox_size = (100, 100)
        bbox = [
            (width // 2 - bbox_size[0] // 2, height // 2 - bbox_size[1] // 2),
            (width // 2 + bbox_size[0] // 2, height // 2 + bbox_size[1] // 2)
        ]
        cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)

        # Extract region of interest (ROI) within the detection box
        roi = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

        if roi.shape[0] > 0 and roi.shape[1] > 0:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            image_tensor = preprocess_frame(roi_rgb)

            # Make prediction
            with torch.no_grad():
                output = model(image_tensor)
                predicted_digit = output.argmax(dim=1).item()

            # Overlay the predicted digit on the frame
            cv2.putText(
                frame,
                f"Predicted Digit: {predicted_digit}",
                (bbox[0][0], bbox[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        # Convert the frame to a Tkinter-compatible image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(image=frame_pil)
        live_feed_label.config(image=frame_tk)
        live_feed_label.image = frame_tk

        # Schedule the next frame update
        root.after(10, update_frame)

    update_frame()

def stop_camera_feed():
    global running
    running = False
    live_feed_label.config(image="")
    live_feed_label.image = None

# GUI setup
root = Tk()
root.title("Digit Recognition")
root.configure(bg="white")

# Set window dimensions
window_width = 1200
window_height = 700
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width // 2) - (window_width // 2)
y_coordinate = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

main_frame = Frame(root, bg="white")
main_frame.pack(fill="both", expand=True)

# Left side for live feed
left_frame = Frame(main_frame, bg="white", width=600, height=700)
left_frame.pack(side="left", fill="both", padx=10, pady=10)
left_frame.pack_propagate(False)

Label(left_frame, text="Live Camera Feed", font=("Helvetica", 16), bg="white").pack(pady=10)
Button(left_frame, text="Start Live Feed", command=start_camera_feed, font=("Helvetica", 14)).pack(pady=5)
Button(left_frame, text="Stop Live Feed", command=stop_camera_feed, font=("Helvetica", 14)).pack(pady=5)

live_feed_label = Label(left_frame, bg="white")
live_feed_label.pack(pady=10, fill="both", expand=True)

# Vertical separator
separator = Frame(main_frame, width=2, bg="black")
separator.pack(side="left", fill="y")

# Right side for upload and analysis
right_frame = Frame(main_frame, bg="white", width=600, height=700)
right_frame.pack(side="left", fill="both", padx=10, pady=10)
right_frame.pack_propagate(False)

Label(right_frame, text="Upload and Analyze Image", font=("Helvetica", 16), bg="white").pack(pady=10)
Button(right_frame, text="Upload Image", command=upload_and_predict, font=("Helvetica", 14)).pack(pady=5)

image_chart_frame = Frame(right_frame, bg="white")
image_chart_frame.pack(pady=10)

preprocessed_image_label = Label(image_chart_frame, bg="white")
preprocessed_image_label.grid(row=0, column=0, padx=10)

# Create a Matplotlib figure for the probability bar chart
fig, ax = plt.subplots(figsize=(2, 2))
canvas = FigureCanvasTkAgg(fig, master=image_chart_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=1, padx=10)

# Initialize the bar chart with zeros
update_bar_chart(ax, canvas, np.zeros(10))

result_label = Label(right_frame, text="Predicted Digit: ", font=("Helvetica", 16), bg="white")
result_label.pack(pady=10)

# Add Quit button
Button(right_frame, text="Quit Program", command=root.quit, font=("Helvetica", 14)).pack(pady=10)

# Start the main GUI loop
root.mainloop()
