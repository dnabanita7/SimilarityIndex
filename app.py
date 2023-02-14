#!/usr/bin/python
from flask import Flask, render_template, Response
import cv2
import face_recognition
import os
import time
from PIL import Image
import numpy as np
from camera import Camera

app = Flask(__name__)

# Load a second sample picture and learn how to recognize it.
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
faces = ""
process_this_frame = True
img = os.path.join('static', 'Image')


def gen(camera):
    while True:
        frame = camera.get_frame()
         # Resize frame of video to 1/4 size for faster face recognition processing
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

            face_names = []
            percentage_similarities = []
            second_face_names = []
            second_percentage_similarities = []
            third_face_names = []
            third_percentage_similarities = []
            for face_encoding in face_encodings:

                # See if the face is a match for the known face(s)

                matches = face_recognition.compare_faces(
                    students_face_encodings, face_encoding
                )
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face

                face_distances = face_recognition.face_distance(
                    students_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                name = students_names[best_match_index]
                similarity = 100 / (1 + face_distances[best_match_index])
                face_names.append(name)
                percentage_similarities.append(similarity)
                # add the second face
                face_distances = np.delete(face_distances, best_match_index)
                second_best_match_index = np.argmin(face_distances)
                second_best_name = students_names[second_best_match_index]
                second_similarity = 100 / (1 + face_distances[second_best_match_index])
                second_face_names.append(second_best_name)
                second_percentage_similarities.append(second_similarity)
                # add the third face
                face_distances = np.delete(face_distances, second_best_match_index)
                third_best_match_index = np.argmin(face_distances)
                third_best_name = students_names[third_best_match_index]
                third_similarity = 100 / (1 + face_distances[third_best_match_index])
                third_face_names.append(third_best_name)
                third_percentage_similarities.append(third_similarity)
                faces = "static/Image/" + face_names[0] + ".png"
                filename = "static/faces.txt"
                # storing the recurring similar faces in a file
                if os.path.exists(filename):
                    append_write = 'a' # append if already exists
                else:
                    append_write = 'w' # make a new file if not

                similar_faces = open(filename,append_write)
                time.sleep(0.01)
                similar_faces.write(faces + '\n')
                similar_faces.close()
                    
                # Display the results
                for (
                    (top, right, bottom, left),
                    name,
                    second_name,
                    third_name,
                    percent,
                    second_percent,
                    third_percent,
                ) in zip(
                    face_locations,
                    face_names,
                    second_face_names,
                    third_face_names,
                    percentage_similarities,
                    second_percentage_similarities,
                    third_percentage_similarities,
                ):
                    text2 = second_name + " with " + str(round(second_percent, 2)) + "%"
                    text3 = third_name + " with " + str(round(third_percent, 2)) + "%"
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(
                        frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
                    )
                    font = cv2.FONT_HERSHEY_DUPLEX
                    text = name + " with " + str(round(percent, 2)) + "%"
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
            
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    files = [os.path.join(img, name) for name in os.listdir(img)]
    with open("static/faces.txt", "r") as file:
        last_line = file.readlines()[-1]
    return render_template("index.html", images=files, last=last_line)

@app.route("/video_feed")
def video_feed():
    return Response(gen(Camera()), mimetype="multipart/x-mixed-replace; boundary=frame")

    #return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)

