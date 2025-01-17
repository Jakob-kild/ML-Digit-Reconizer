# Digit Recognition Project

## Overview
This project implements a real-time digit recognition application using a pre-trained neural network. The GUI allows users to:
- Start a live camera feed to detect handwritten digits within a bounding box.
- Upload an image and analyze the detected digits.

The project uses PyTorch for the main model and TensorFlow for compatibility with an alternative model.

## Features
- Real-time digit detection in a bounding box using live video.
- Analyze uploaded images for digit recognition.
- Visualize the confidence scores using a bar chart.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Jakob-kild/ML-Digit-Reconizer.git
   cd ML-Digit-Reconizer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the application:
   ```bash
   python main.py
   ```

2. **Live Feed**:
   - Click "Start Live Feed" to begin the camera.
   - Place a handwritten digit within the bounding box to detect it.

3. **Upload Image**:
   - Click "Upload Image" to select an image file for analysis.

4. Quit the program using the "Quit Program" button.

## Project Structure
- `main.py`: Main script for the GUI and application logic.
- `model/`: Folder containing the pre-trained models.
- `src/train_model.py`: Script for the model definition.

## Requirements
Refer to the `requirements.txt` file for dependencies.

## License
This project is open-source and available under the [MIT License](LICENSE).
