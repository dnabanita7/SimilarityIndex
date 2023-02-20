#!/usr/bin/python
from flask import Flask, render_template, Response, send_file
import cv2
import face_recognition
import os
import time
from PIL import Image
import numpy as np
from camera import Camera

app = Flask(__name__)

students_face_encodings = []
for i in range(43):
    content = face_recognition.load_image_file(
        "static/Image/students" + str(i).zfill(2) + ".png"
    )
    student_face_encodings = face_recognition.face_encodings(content)[0]
    students_face_encodings.append(student_face_encodings)

# Create arrays of known face encodings and their names
students_names = ["students" + str(i).zfill(2) for i in range(43)]
# Initialize some variables
face_locations = []
face_encodings = []
student_match = ""

img = os.path.join('static', 'Image')

def find_match(student_face_encodings, face_encoding, face_distances, matches, position):
    match_index = np.argmin(face_distances)
    if (position != 0):
        face_distances = np.delete(face_distances, match_index)
        match_index = np.argmin(face_distances)

    name = students_names[match_index]
    similarity = 100 / (1 + face_distances[match_index])
    matches[position] = {"name": name, "similarity": similarity}
    return matches, face_distances

def gen(camera):
    process_this_frame = False
    while True:
        process_this_frame = True if process_this_frame == False else False
        frame = camera.get_frame()

        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            matches = {}

            for face_encoding in face_encodings:
                face_distances = face_recognition.face_distance(
                    students_face_encodings, face_encoding
                )
                matches,face_distances = find_match(student_face_encodings, face_encoding, face_distances, matches, 0)
                matches,face_distances = find_match(student_face_encodings, face_encoding, face_distances, matches, 1)
                matches,face_distances = find_match(student_face_encodings, face_encoding, face_distances, matches, 2)

                top, right, bottom, left = face_locations[0]
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                font = cv2.FONT_HERSHEY_DUPLEX
                text = matches[0]["name"] + " with " + str(round(matches[0]["similarity"], 2)) + "%"
                text2 = matches[1]["name"] + " with " + str(round(matches[1]["similarity"], 2)) + "%"
                text3 = matches[2]["name"] + " with " + str(round(matches[2]["similarity"], 2)) + "%"

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(
                    frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
                )

                cv2.putText(
                    frame, text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
                )

                cv2.putText(
                    frame,
                    text2,
                    (left + 6, bottom + 15),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (0xFF, 0xFF, 0),
                    1,
                )

                cv2.putText(
                    frame,
                    text3,
                    (left + 6, bottom + 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (0xFF, 0xFF, 0),
                    1,
                )
                student_match = f"static/Image/{matches[0]['name']}.png"

                f = open('static/faces.txt', 'w')
                f.write(student_match)
                cv2.imwrite('static/match.png', cv2.imread(student_match))

        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    images = [os.path.join(img, name) for name in os.listdir(img)]
    names = open("static/names.txt").readlines()
    family_name = []
    first_name = []
    for name in names:
        name = name[0:-1]
        family, first = name.split(":")
        family_name.append(family)
        first_name.append(first)

    return render_template("index.html", images=zip(images, family_name, first_name), last=student_match)

@app.route("/video_feed")
def video_feed():
    return Response(gen(Camera()), mimetype="multipart/x-mixed-replace; boundary=frame")

def gen_student():
    while True:
        f = open("static/faces.txt", 'r')
        student = f.read()
        if len(student) >= 2:
            image = cv2.imread(student, flags=cv2.IMREAD_COLOR)
            frame = cv2.imencode('.png', image)[1].tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
        yield (b'')

@app.route('/image_feed')
def image_feed():
    return Response(gen_student(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
