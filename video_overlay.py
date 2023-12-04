import cv2
import numpy as np

def show_video(idx):
    # Read the videos
    video1 = cv2.VideoCapture(f'./recordings/human-episode-{idx}.mp4')
    video2 = cv2.VideoCapture(f'./recordings/agent-episode-{idx}.mp4')

    # Get properties of video1
    width = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video1.get(cv2.CAP_PROP_FPS)

    last_frame_1 = None
    last_frame_2 = None

    # Loop through frames
    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        if ret1:
            last_frame_1 = frame1
        if ret2:
            last_frame_2 = frame2

        if not ret1 and not ret2:
            break

        if not ret1:
            frame1 = last_frame_1
        if not ret2:
            frame2 = last_frame_2

        # Resize frame2 to match the dimensions of frame1
        frame2 = cv2.resize(frame2, (width, height))
        # car = np.zeros_like(frame2)
        # car[int(car.shape[0] / 2 - 25): int(car.shape[0] / 2 + 25), int(car.shape[1] / 2 - 25): int(car.shape[1] / 2 + 25)] = frame2[int(car.shape[0] / 2 - 25): int(car.shape[0] / 2 + 25), int(car.shape[1] / 2 - 25): int(car.shape[1] / 2 + 25)]

        # Extract the alpha channel from the second video
        # b, g, r = cv2.split(car)
        b, g, r = cv2.split(frame2)

        # Perform alpha blending
        overlay = cv2.merge((b, g, r))
        background = frame1
        composite = cv2.addWeighted(overlay, 0.5, background, 0.5, 0, background)

        cv2.imshow('Video', composite) 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # Release video capture and writer
    video1.release()
    video2.release()
    cv2.destroyAllWindows()

