#!/usr/bin/python
from flask import Flask, render_template, Response, jsonify
import cv2
import face_recognition
import os
import time
from PIL import Image
import numpy as np
from camera import Camera

app = Flask(__name__)

FONT = cv2.FONT_HERSHEY_DUPLEX

def setup():
    """
    Creates student_faces a list that contains a json object for each student.

    This object contains the student's name, image and facial encodings.

    Parameters
    ----------
    None

    Returns
    -------
    list
    students_faces
    A list of objects containing information about each student.
    """
    students_faces = []
    names = open("static/names.txt").readlines()
    for i in range(43):
        student = {}

        name = names[i]
        name = name[0:-1]
        family, first = name.split(":")
        student['family_name'] = family
        student['first_name'] = first

        image_url = f"static/Image/students{str(i).zfill(2)}.png"
        student['img'] = image_url

        content = face_recognition.load_image_file(image_url)
        student_face_encodings = face_recognition.face_encodings(content)[0]
        student['face_encoding'] = student_face_encodings

        students_faces.append(student)

    return students_faces

def find_match(face):
    """
    Finds the match for the passed face

    This function then deletes that match from the available matches for the
    passed face. So if called twice in a row it will be finding first the
    best match and then the second best match.

    Parameters
    ----------
    face : dict
    The face we're finding a match for

    Returns
    -------
    dict
        With a value added to it's "matches" list.
    """
    match_index = np.argmin(face['face_distances'])

    name = f"{students_faces[match_index]['first_name']} {students_faces[match_index]['family_name']}"
    image = students_faces[match_index]['img']
    similarity = 100 / (1 + face['face_distances'][match_index])
    match = {"name": name, "similarity": similarity, "img": image}

    face['matches'].append(match)
    face['face_distances'] = np.delete(face['face_distances'], match_index)
    return face

def get_similarity_string(student_name, similarity_percentage):
    """
    Returns the string we want to display about how similar someone is to
    the student

    Parameters
    ----------
    student_name : string
    The name of the student we've matched to

    similarity_percentage: string
    The amount of similarity we share with the student

    Returns
    -------
    String
        To be displayed on the frontend

    """
    return f"{student_name} with {str(round(similarity_percentage, 2))} %"

def get_face_positions(face_position):
    """
    Returns the adjusted top,right,bottom,left positions

    Parameters
    ----------
    face_position : tuple
    The original positions of the face

    Returns
    -------
    tuple

    """
    top, right, bottom, left = face_position
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
    return top, right, bottom, left

def draw_other_match_info(frame, seen_face, face_position):
    """
    Draws information about the second and third closest student on the screen.

    Parameters
    ----------
    frame : image data
    seen_face : dict
    face_position: tuple
    The name of the student we've matched to

    similarity_percentage: string
    The amount of similarity we share with the student

    Returns
    -------
    String
        To be displayed on the frontend
    """
    # Fetch the adjusted face position
    top, right, bottom, left = get_face_positions(face_position)

    # Draw a label with a name below the face
    cv2.rectangle(
        frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
    )

    # Write out information about the second closest student
    text2 = get_similarity_string(seen_face['matches'][1]['name'], seen_face['matches'][1]['similarity'])
    cv2.putText(
        frame,
        text2,
        (left + 6, bottom + 15),
        FONT,
        0.5,
        (0xFF, 0xFF, 0),
        1,
    )
    # Write out information about the third closest student
    text3 = get_similarity_string(seen_face['matches'][2]['name'], seen_face['matches'][2]['similarity'])
    cv2.putText(
        frame,
        text3,
        (left + 6, bottom + 30),
        FONT,
        0.5,
        (0xFF, 0xFF, 0),
        1,
    )

def draw_main_match_info(frame, seen_face, face_position):
    """
    Draws information about the closest student on the screen.
    """
    top, right, bottom, left = get_face_positions(face_position)

    text = get_similarity_string(seen_face['matches'][0]['name'], seen_face['matches'][0]['similarity'])

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    cv2.putText(
        frame, text, (left + 6, bottom - 6), FONT, 1.0, (255, 255, 255), 1
    )

def gen(camera):
    # Give this an intial value
    process_this_frame = False

    while True:
        # Use ternary operator to flip this ever call
        process_this_frame = True if process_this_frame == False else False

        # Get the most current name from the camera
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

            # Create a list of the available student face encodings
            students_face_encodings = [key['face_encoding'] for key in students_faces]

            # For the visible faces
            for i in range(0, len(face_encodings)):

                # Construct a face object.
                seen_face = {}

                # Give the face it's encoding and location
                seen_face['face_encoding'] = face_encodings[i]
                seen_face['face_locations'] = face_locations[i]

                # Calculate and assign it's distance from the student faces
                face_distances = face_recognition.face_distance(
                    students_face_encodings, face_encodings[i]
                )
                seen_face['face_distances'] = face_distances

                # Before we find the exact matches create it's match list
                seen_face['matches'] = []

                # Assign the face it's top three matches
                seen_face = find_match(seen_face)
                seen_face = find_match(seen_face)
                seen_face = find_match(seen_face)

                # Draw what we want to draw regarding the best match
                draw_main_match_info(frame, seen_face, face_locations[0])

                # Draw what we want to draw regarding other matches
                draw_other_match_info(frame, seen_face, face_locations[0])

                # Write out the best match to the faces.txt file
                f = open('static/faces.txt', 'w')
                f.write(f"{seen_face['matches'][0]['img']}")

        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_student():
    while True:
        f = open("static/faces.txt", 'r')
        student = f.read()
        if len(student) >= 2:
            f = cv2.imencode('.jpg', cv2.imread(str(student.strip())))[1].tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' + f + b'\r\n')

def matched_student():
    while True:
        f = open("static/faces.txt", "r")
        student = f.read()
        if len(student) >= 2:
            student_number = int(student[-6:-5])
            matched_name = students_faces[student_number]['first_name'] + " " + students_faces[student_number]['family_name']
            yield matched_name

@app.route("/")
def index():
    return render_template("index.html", students_faces=students_faces)

@app.route("/video_feed")
def video_feed():
    return Response(gen(Camera()), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/image_feed')
def image_feed():
    return Response(gen_student(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/matched_name")
def matched_name():
    return Response(matched_student(), mimetype="text/plain")


if __name__ == "__main__":
    students_faces = setup()
    app.run(debug=True)
