#!/usr/bin/python
# -*- coding: utf-8 -*-
import face_recognition
import cv2
import numpy as np

# Get a reference to webcam #0 (the default one)

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.

students_face_encodings = []
for i in range(43):
    content = face_recognition.load_image_file(
        "portraits/students" + str(i).zfill(2) + ".png"
    )
    student_face_encodings = face_recognition.face_encodings(content)[0]
    students_face_encodings.append(student_face_encodings)

# Create arrays of known face encodings and their names

students_names = ["student" + str(i).zfill(2) for i in range(43)]

# Initialize some variables

face_locations = []
face_encodings = []
face_names = []
percentage_similarities = []
process_this_frame = True

while True:

    # Grab a single frame of video

    (ret, frame) = video_capture.read()

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
            face_distances = np.delete(face_distances, best_match_index)
            second_best_match_index = np.argmin(face_distances)
            second_best_name = students_names[second_best_match_index]
            second_similarity = 100 / (1 + face_distances[second_best_match_index])
            face_distances = np.delete(face_distances, second_best_match_index)
            third_best_match_index = np.argmin(face_distances)
            third_best_name = students_names[third_best_match_index]
            third_similarity = 100 / (1 + face_distances[third_best_match_index])
    process_this_frame = not process_this_frame

    # Display the results

    for ((top, right, bottom, left), name, percent) in zip(
        face_locations, face_names, percentage_similarities
    ):
        text2 = second_best_name + " with " + str(round(second_similarity, 2)) + "%"
        text3 = third_best_name + " with " + str(round(third_similarity, 2)) + "%"

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0xFF), 2)

        # Draw a label with a name below the face

        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 0xFF), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        text = name + " with " + str(round(percent, 2)) + "%"
        cv2.putText(
            frame,
            text,
            (left + 6, bottom - 6),
            font,
            0.5,
            (0xFF, 0xFF, 0xFF),
            1,
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

    # Display the resulting image

    cv2.imshow("Video", frame)

    # Hit 'q' on the keyboard to quit!

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to the webcam

video_capture.release()
cv2.destroyAllWindows()

