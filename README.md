# Pixel Art Editor with MNIST Digit Prediction

This project combines a pixel art editor and a digit recognition model trained on the MNIST dataset. Users can draw pixel art and have the model predict the digit they have drawn. The prediction is displayed as a histogram of probabilities for each digit from 0 to 9.

## Features

- **Pixel Art Editor**: A grid-based drawing interface where users can draw using a pencil or erase using an eraser.
- **MNIST Digit Recognition**: A pre-trained Convolutional Neural Network (CNN) model that predicts the digit drawn by the user.
- **Real-time Prediction**: The model predicts the digit and updates a histogram displaying the prediction probabilities every 1.5 seconds.
- **Histogram Visualization**: Displays the probabilities of each digit (0-9) based on the user's drawing.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Pygame
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/pixel-art-mnist.git
    cd pixel-art-mnist
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the MNIST model file (`model.h5`) saved in the correct directory as specified in the code (`C:\Users\user\Desktop\Partfolio GPT\01\model.h5`). If you do not have the model file, follow the steps below to train and save it.

## Training the MNIST Model

If you need to train the MNIST model from scratch:

1. Run the training script to train the model and save it:
    ```python
    python train_model.py
    ```

## Usage

1. Start the pixel art editor:
    ```python
    python test_model.py
    ```

2. Draw on the grid using the mouse:
    - Left-click to draw with the pencil.
    - Press `P` to switch to pencil mode.
    - Press `E` to switch to eraser mode.
    - Press `C` to clear the grid.
    - Press `S` to convert the drawing to grayscale.

3. The histogram on the right side of the window will update every 1.5 seconds with the predicted probabilities for each digit (0-9).

## File Structure

- `pixel_art_editor.py`: Main application code for the pixel art editor and digit prediction.
- `train_mnist_model.py`: Script to train the MNIST model and save it as `model.h5`.
- `requirements.txt`: List of required Python packages.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to customize this README further based on your specific repository structure and needs.
