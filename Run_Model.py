import tensorflow as tf
import numpy as np
import cv2
import pygame

# Load the trained model
model = tf.keras.models.load_model("model.keras")

# Initialize pygame
pygame.init()
width, height = 500,500
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Draw a Number")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Initialize canvas
screen.fill(BLACK)


def preprocess_image():
    data = pygame.image.tostring(screen, "RGB")
    img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find the bounding box of the drawn digit
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(cv2.convexHull(np.vstack(contours)))
        img = img[y:y + h, x:x + w]  # Crop to bounding box

        # Resize while maintaining aspect ratio
        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w, new_h = 20, int(20 / aspect_ratio)
        else:
            new_w, new_h = int(20 * aspect_ratio), 20
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Place the resized digit in the center of a 28x28 black image
        padded_img = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img
        img = padded_img
    else:
        img = np.zeros((28, 28), dtype=np.uint8)  # If nothing is drawn

    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply slight blur to smooth edges
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def predict_number():
    img = preprocess_image()
    prediction = model.predict(img)
    return np.argmax(prediction)


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONUP:  # Detect mouse release
            predicted_number = predict_number()
            pygame.display.set_caption(f"Predicted: {predicted_number}")
        elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:  # Left mouse button to draw
            pos = pygame.mouse.get_pos()
            pygame.draw.circle(screen, WHITE, pos, 10)  # Reduced brush size for precision

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:  # Clear the drawing
        screen.fill(BLACK)

    pygame.display.flip()
    clock.tick(100)

pygame.quit()
