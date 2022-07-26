# Program Constants
num_hands = 1  # number of hands that the program will detect at one time
num_class = 6  # number of different hand classes to detect
num_instance = 500  # training size
break_time = 2  # min of 2 seconds

if __name__ == '__main__':
    # Required Library Imports
    from cv2 import cv2
    import pandas as pd
    import time

    # Importing Helper Classes
    import detect_hands
    from make_calculations import calculate
    from helper import classify_class

    full_data = []
    data_target = 0

    hands = detect_hands.hand_detector(max_hands=num_hands)
    cap = cv2.VideoCapture(0)
    print("Now collecting data for: '0'", "\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        image, list = hands.find_hand_landmarks(
            cv2.flip(frame, 1), draw_landmarks=True)

        if list:
            height, width, _ = image.shape
            distance_list = calculate(height, width, list)
            full_data.append(distance_list)
            print(len(full_data))

        cv2.imshow('Hands', image)
        cv2.waitKey(2)

        if len(full_data) >= num_instance:
            print('Creating Pandas DataFrame...', )
            hand1_df = pd.DataFrame(full_data)
            hand1_df['y'] = data_target
            hand1_df.to_csv(f'hand-{data_target}.csv', index=False)
            data_target += 1
            full_data = []
            if data_target >= num_class:
                break
            else:
                print('Get ready to train the next class .... ' +
                      classify_class(data_target))
                time.sleep(break_time)

    cap.release()
