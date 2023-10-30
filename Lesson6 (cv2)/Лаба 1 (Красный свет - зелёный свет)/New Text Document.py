import cv2
import numpy as np


def get_decorated_frame(curr_frame, is_redlight, is_moving):
    cv2.rectangle(curr_frame, (0, curr_frame.shape[0]-61), (curr_frame.shape[1]-1, curr_frame.shape[0]-1), (37, 37, 44), -1)

    if is_redlight:
        cv2.circle(curr_frame, (curr_frame.shape[1]//2 +32, curr_frame.shape[0]-31), 20, (0, 0, 255), -1)
        cv2.putText(curr_frame, "KPACHblu CBET", (curr_frame.shape[1]//2 +62, curr_frame.shape[0]-21), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if is_moving:
            cv2.rectangle(curr_frame, (0, curr_frame.shape[0]-61), (curr_frame.shape[1]//2-1, curr_frame.shape[0]-1), (0, 0, 255), -1)
        else:
            cv2.rectangle(curr_frame, (0, curr_frame.shape[0]-61), (curr_frame.shape[1]//2-1, curr_frame.shape[0]-1), (0, 255, 0), -1)

    else:
        cv2.circle(curr_frame, (curr_frame.shape[1]//2 +32, curr_frame.shape[0]-31), 20, (0, 255, 0), -1)
        cv2.putText(curr_frame, "3eJleHblu CBET", (curr_frame.shape[1]//2 +62, curr_frame.shape[0]-21), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return curr_frame


webcam = cv2.VideoCapture(0)
tick = 1 # для псвевдослучайного таймера красного/зелёного света
prev_frame = cv2.cvtColor(webcam.read()[1], cv2.COLOR_BGR2GRAY)

while True:
    key = cv2.waitKey(20)
    frame_raw = webcam.read()[1]

    if key == 27: # Esc
        break
    else:
        tick = tick % 500 + np.random.randint(1, 10)
        is_redlight = tick < 350

        frame = cv2.medianBlur(frame_raw, 15)
        frame = cv2.adaptiveThreshold( # doing things the easy way
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            15, 3
        )
        frame_diff = cv2.absdiff(frame, prev_frame)
        is_moving = (cv2.countNonZero(frame_diff) / frame_diff.size) > 0.01
        prev_frame = frame

        if is_redlight and is_moving:
            frame_raw_controus = cv2.drawContours(
                frame_raw,
                cv2.findContours(
                    frame_diff,
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE
                )[0],
                -1, (0, 0, 155), 5
            )
            decorated_frame = get_decorated_frame(frame_raw_controus, is_redlight, is_moving)
        else:
            decorated_frame = get_decorated_frame(frame_raw, is_redlight, is_moving)

        cv2.imshow('KAMEPA', decorated_frame)


cv2.destroyAllWindows()
webcam.release()
