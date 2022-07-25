# Required Library Imports
from helper import classify_class
from make_calculations import calculate
from generate_data import num_hands
import detect_hands
from cv2 import cv2
import pickle
import warnings
import random
import time
warnings.filterwarnings("ignore")


class Game:
    def __init__(self):
        # Importing Helper Classes

        # Settings For The Text
        self.coordinates = (10, 30)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.75
        self.color = (168, 108, 1)  # blue-green-red
        self.thickness = 2
        self.type = cv2.LINE_AA
        self.hands = detect_hands.hand_detector(max_hands=num_hands)
        self.model = pickle.load(open('model.sav', 'rb'))
        self.cap = cv2.VideoCapture(0)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.timer = 0
        self.prediction = None
        self.computer_hand = None
        self.prediction_text = None
        self.comp_prediction_text = None
        self.playerScore = 0
        self.compScore = 0

    def splash_screen(self):
        text1 = "Welcome to Hand Cricket!"
        textsize1 = cv2.getTextSize(text1, self.font, 2, 3)[0]
        textX1 = int((self.width / 2) - (textsize1[0] / 2))
        textY1 = int((self.height / 2) - (textsize1[1] / 2))

        coin = random.randint(0,1)
        if coin == 0:
            first = "BAT"
        else:
            first = "BALL"

        text2 = "Based on a random coin toss, you will {} first!".format(first)
        textsize2 = cv2.getTextSize(text2, self.font, 1, 2)[0]
        textX2 = int((self.width / 2) - (textsize2[0] / 2))
        textY2 = int((self.height / 2) - (textsize2[1] / 2))

        text3 = "Keep your hand in the frame to play! Game pauses when you move it out. Press 'q' to quit."
        textsize3 = cv2.getTextSize(text3, self.font, 0.75, 2)[0]
        textX3 = int((self.width / 2) - (textsize3[0] / 2))
        textY3 = int((self.height / 2) - (textsize3[1] / 2))

        text4 = "Press space to start!"
        textsize4 = cv2.getTextSize(text4, self.font, 0.75, 2)[0]
        textX4 = int((self.width / 2) - (textsize4[0] / 2))
        textY4 = int((self.height / 2) - (textsize4[1] / 2))
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image, list = self.hands.find_hand_landmarks(
                cv2.flip(frame, 1),
                draw_landmarks=False
            )
            cv2.putText(
                image,
                text1,
                (textX1, textY1-30),
                self.font,
                2,
                (58, 138, 79),
                3,
                self.type
            )
            cv2.putText(
                image,
                text2,
                (textX2, textY2 + 20),
                self.font,
                1,
                (60, 207, 202),
                self.thickness,
                self.type
            )
            cv2.putText(
                image,
                text3,
                (textX3, textY3 + 80),
                self.font,
                0.75,
                (212, 83, 19),
                self.thickness,
                self.type
            )
            cv2.putText(
                image,
                text4,
                (textX4, textY4 + 140),
                self.font,
                0.75,
                (10, 59, 252),
                self.thickness,
                self.type
            )

            cv2.imshow('Hand Cricket Instructions', image)
            self.timer = self.timer + 1

            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
        self.run_game(coin)

    def run_game(self, coin):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image, list = self.hands.find_hand_landmarks(
                cv2.flip(frame, 1),
                draw_landmarks=False
            )

            if list:
                height, width, _ = image.shape
                all_distance = calculate(height, width, list)
                if self.timer % 30 == 0 and self.timer > 0:
                    self.prediction = self.model.predict([all_distance])[0]
                    self.computer_hand = random.choice([0, 1, 2, 3, 4, 6])
                    self.prediction_text = "You chose " + \
                        classify_class(self.prediction)
                    self.comp_prediction_text = "The computer chose " + \
                        classify_class(self.computer_hand)

                if (self.prediction == 0 and self.computer_hand == 1) or (self.prediction == 1 and self.computer_hand == 2) or (
                        self.prediction == 2 and self.computer_hand == 0):
                    if self.timer % 60 == 0 and self.timer > 0:
                        self.playerScore = self.playerScore + 1
                    cv2.putText(
                        image,
                        "You Win!",
                        (600, 250),
                        self.font,
                        self.fontScale,
                        (3, 191, 8),
                        self.thickness,
                        self.type
                    )
                elif (self.prediction == 1 and self.computer_hand == 0) or (self.prediction == 2 and self.computer_hand == 1) or (
                        self.prediction == 0 and self.computer_hand == 2):
                    if self.timer % 60 == 0 and self.timer > 0:
                        self.compScore = self.compScore + 1
                    cv2.putText(
                        image,
                        "You Lose!",
                        (600, 250),
                        self.font,
                        self.fontScale,
                        (10, 45, 176),
                        self.thickness,
                        self.type
                    )
                elif(self.prediction == self.computer_hand and self.prediction != None and self.computer_hand != None):
                    cv2.putText(
                        image,
                        "Draw!",
                        (600, 250),
                        self.font,
                        self.fontScale,
                        (176, 10, 19),
                        self.thickness,
                        self.type
                    )
                cv2.putText(
                    image,
                    "Computer Score: " + str(self.compScore),
                    (10, 400),
                    self.font,
                    self.fontScale,
                    (168, 108, 1),
                    self.thickness,
                    self.type
                )
                cv2.putText(
                    image,
                    "Player Score: " + str(self.playerScore),
                    (1080, 400),
                    self.font,
                    self.fontScale,
                    (16, 210, 236),
                    self.thickness,
                    self.type
                )
                image = cv2.putText(
                    image,
                    self.comp_prediction_text,
                    self.coordinates,
                    self.font,
                    self.fontScale,
                    self.color,
                    self.thickness,
                    self.type
                )
                cv2.putText(
                    image,
                    self.prediction_text,
                    (1030, 30),
                    self.font,
                    self.fontScale,
                    (16, 210, 236),
                    self.thickness,
                    self.type
                )
                cv2.putText(
                    image,
                    str(((30 - int(self.timer % 30)) // 10) + 1),
                    (650, 30),
                    self.font,
                    self.fontScale,
                    (10, 45, 176),
                    self.thickness,
                    self.type
                )
            cv2.imshow('Hands', image)
            self.timer = self.timer + 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    # while cap.isOpened():
    #     now = time.time()
    #     success, frame = cap.read()
    #     if not success:
    #         print("Ignoring empty camera frame.")
    #         continue
    #     image, list = hands.find_hand_landmarks(
    #         cv2.flip(frame, 1),
    #         draw_landmarks=False
    #     )

    #     if time.time() - now < 4:
    #         cv2.putText(image, "Time For the Toss!", (650, 30),
    #                     font, fontScale, color, thickness, self.type)


game = Game()

game.splash_screen()
