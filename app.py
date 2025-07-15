from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import math
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def detect_balls(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=10, maxRadius=40
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Return list of (x,y,radius)
        return circles
    else:
        return []

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    input_path = os.path.join(UPLOAD_FOLDER, 'input.jpg')
    file.save(input_path)

    circles = detect_balls(input_path)

    if len(circles) < 2:
        return "Need at least two balls detected for calculation. Try another image.", 400

    # For demo: take first circle as cue ball, second as target ball
    cue_ball = (circles[0][0], circles[0][1])
    target_ball = (circles[1][0], circles[1][1])

    angle = calculate_angle(cue_ball, target_ball)

    # Draw circles and line on the image
    img = cv2.imread(input_path)

    # Draw cue ball in green
    cv2.circle(img, cue_ball, circles[0][2], (0,255,0), 3)
    # Draw target ball in red
    cv2.circle(img, target_ball, circles[1][2], (0,0,255), 3)
    # Draw line between cue ball and target ball in blue
    cv2.line(img, cue_ball, target_ball, (255,0,0), 2)

    # Put angle text on image
    cv2.putText(
        img, f'Shoot Angle: {angle:.1f} deg',
        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
    )

    output_path = os.path.join(UPLOAD_FOLDER, 'output.jpg')
    cv2.imwrite(output_path, img)

    # Render template with result image and angle
    return render_template('result.html', angle=angle, image_file='output.jpg')

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)