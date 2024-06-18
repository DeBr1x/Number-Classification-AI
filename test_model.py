import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.models import load_model  # Suppress TF logs # type: ignore
import tensorflow as tf
import pygame
import sys
import numpy as np

# Constants
WINDOW_SIZE = 560
GRID_SIZE = 28
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ERASER_COLOR = WHITE
HISTOGRAM_BINS = 10
MODEL_PATH = r'C:\Users\user\Desktop\Partfolio GPT\01\model.h5'

# Initialize Pygame
pygame.init()

# Setup display
screen = pygame.display.set_mode((WINDOW_SIZE * 2, WINDOW_SIZE))
pygame.display.set_caption('Pixel Art Editor')

# Create histogram surface
histogram_window = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))

# Initialize color matrix and histogram
color_matrix = [[WHITE for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
histogram = [0] * HISTOGRAM_BINS

# Tools
tool = 'pencil'
current_color = BLACK

# Load the model
model = load_model(MODEL_PATH)

def draw_grid(surface):
    """Draw the grid on the given surface."""
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        for y in range(0, WINDOW_SIZE, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, BLACK, rect, 1)

def draw_pixels(surface):
    """Draw the pixels on the given surface based on color_matrix."""
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color_matrix[y][x], rect)

def draw_histogram(surface):
    """Draw the histogram on the given surface."""
    surface.fill(WHITE)
    max_value = max(histogram)
    if max_value == 0:
        return
    bin_width = WINDOW_SIZE // HISTOGRAM_BINS
    for i in range(HISTOGRAM_BINS):
        bin_height = (histogram[i] / max_value) * (WINDOW_SIZE - 20)
        pygame.draw.rect(surface, BLACK, (i * bin_width, WINDOW_SIZE - bin_height, bin_width, bin_height))
        font = pygame.font.Font(None, 20)
        text = font.render(str(i), True, (128, 128, 128))
        text_rect = text.get_rect(center=(i * bin_width + bin_width // 2, WINDOW_SIZE - 10))
        surface.blit(text, text_rect)

def get_cell(x, y):
    """Get the grid cell coordinates for a given x, y position."""
    return x // CELL_SIZE, y // CELL_SIZE

def handle_input():
    """Handle user input events."""
    global tool, current_color, color_matrix

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                tool = 'pencil'
                current_color = BLACK
            elif event.key == pygame.K_e:
                tool = 'eraser'
                current_color = ERASER_COLOR
            elif event.key == pygame.K_c:
                color_matrix = [[WHITE for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
            elif event.key == pygame.K_s:
                update_color_matrix_from_brightness()

def update_matrix(mouse_pos):
    """Update the color matrix based on the mouse position and selected tool."""
    x, y = get_cell(*mouse_pos)
    for dy in range(2):
        for dx in range(2):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if tool == 'pencil':
                    color_matrix[ny][nx] = current_color
                elif tool == 'eraser':
                    color_matrix[ny][nx] = ERASER_COLOR

def calculate_brightness(color):
    """Calculate the brightness of a given color."""
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

def update_color_matrix_from_brightness():
    """Update the color matrix to grayscale based on brightness."""
    global color_matrix
    brightness_matrix = [[calculate_brightness(color_matrix[y][x]) for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            brightness = brightness_matrix[y][x]
            color_matrix[y][x] = (brightness, brightness, brightness)

def update_histogram(arr):
    """Update the histogram with new values."""
    global histogram
    histogram = arr

def convert_to_mnist_format():
    """Convert the color matrix to the MNIST format expected by the model."""
    img = np.zeros((GRID_SIZE, GRID_SIZE))
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            img[y][x] = 1 if color_matrix[y][x] == BLACK else 0
    img = img.reshape((1, GRID_SIZE, GRID_SIZE, 1))
    return img

def predict_digit():
    """Predict the digit drawn on the grid."""
    img = convert_to_mnist_format()
    prediction = model.predict(img)
    histogram = prediction[0] * 100
    return histogram

def main():
    """Main game loop."""
    clock = pygame.time.Clock()
    last_prediction_time = pygame.time.get_ticks()

    while True:
        handle_input()

        if pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            update_matrix(mouse_pos)

        screen.fill(WHITE)
        histogram_window.fill(WHITE)

        draw_pixels(screen)
        draw_grid(screen)

        current_time = pygame.time.get_ticks()
        if current_time - last_prediction_time >= 1500:
            update_histogram(predict_digit())
            last_prediction_time = current_time

        draw_histogram(histogram_window)
        screen.blit(histogram_window, (WINDOW_SIZE, 0))

        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()
