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

class Player:
    def __init__(self):
        self.score = 0

    def add_score(self, score):
        self.score += score

class Computer:
    def __init__(self):
        self.score = 0

    def add_score(self, score):
        self.score += score

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
        self.player = Player()
        self.computer = Computer()

    def splash_screen(self):
        text1 = "Welcome to Hand Cricket!"
        textsize1 = cv2.getTextSize(text1, self.font, 2, 3)[0]
        textX1 = int((self.width / 2) - (textsize1[0] / 2))
        textY1 = int((self.height / 2) - (textsize1[1] / 2))

        coin = random.randint(0,1)
        if coin == 0:
            first = "BAT"
        else:
            first = "BOWL"

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
            timer = timer + 1

            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        if coin == 0:
            self.bat()
        else:
            self.bowl()

    def run_game(self, coin):
        player_move = 0
        player_text = ""
        computer_move = 0
        computer_text = ""
        timer = 0
        second_innings = False

        if coin == 0:
            first = self.player
            second = self.computer
        else:
            first = self.computer
            second = self.player

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
                
                if timer % 30 == 0 and timer > 0:
                    player_move = self.model.predict([all_distance])[0]
                    computer_move = random.choice([1])
                    player_text = "You chose " + \
                        classify_class(player_move)
                    computer_text = "The computer chose " + \
                        classify_class(computer_move)

                    if player_move == computer_move:
                        if not second_innings:
                            print(self.player.score, self.computer.score)
                            second_innings = True
                            self.end_innings()

                        first, second = second, first
                        # self.loser()
                    else:
                        self.bat(first, player_move)
                    

                cv2.putText(
                    image,
                    "Computer Score: " + str(self.computer.score),
                    (10, 400),
                    self.font,
                    self.fontScale,
                    (168, 108, 1),
                    self.thickness,
                    self.type
                )
                cv2.putText(
                    image,
                    "Player Score: " + str(self.player.score),
                    (1080, 400),
                    self.font,
                    self.fontScale,
                    (17, 75, 212),
                    self.thickness,
                    self.type
                )
                image = cv2.putText(
                    image,
                    computer_text,
                    self.coordinates,
                    self.font,
                    self.fontScale,
                    self.color,
                    self.thickness,
                    self.type
                )
                cv2.putText(
                    image,
                    player_text,
                    (1030, 30),
                    self.font,
                    self.fontScale,
                    (17, 75, 212),
                    self.thickness,
                    self.type
                )
                cv2.putText(
                    image,
                    str(3 - (int(timer % 30) // 10)),
                    (650, 30),
                    self.font,
                    self.fontScale,
                    (10, 45, 176),
                    self.thickness,
                    self.type
                )
            cv2.imshow('Play Cricket!', image)
            timer = timer + 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def bat(self, batter, runs):
        if batter == self.player:
            self.player.add_score(runs)
        else:
            self.computer.add_score(runs)

    def end_innings(self):
        text = "You Win! :)"
        textsize = cv2.getTextSize(text, self.font, 3, 4)[0]
        textX = int((self.width / 2) - (textsize[0] / 2))
        textY = int((self.height / 2) - (textsize[1] / 2))
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
                text,
                (textX, textY),
                self.font,
                3,
                (58, 138, 79),
                4,
                self.type
            )
            if cv2.waitKey(1) & 0xFF == ord(' '):
                return
            cv2.imshow('Thanks for playing!', image)

        cv2.destroyWindow("Thanks for playing!")


    def winner(self):
        text = "You Win! :)"
        textsize = cv2.getTextSize(text, self.font, 3, 4)[0]
        textX = int((self.width / 2) - (textsize[0] / 2))
        textY = int((self.height / 2) - (textsize[1] / 2))
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
                text,
                (textX, textY),
                self.font,
                3,
                (58, 138, 79),
                4,
                self.type
            )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
            cv2.imshow('Thanks for playing!', image)

        self.cap.release()
        cv2.destroyAllWindows()


    def loser(self):
        text = "Better Luck Next Time :("
        textsize = cv2.getTextSize(text, self.font, 3, 4)[0]
        textX = int((self.width / 2) - (textsize[0] / 2))
        textY = int((self.height / 2) - (textsize[1] / 2))
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
                text,
                (textX, textY),
                self.font,
                3,
                (10, 59, 252),
                4,
                self.type
            )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
            cv2.imshow('Thanks for playing!', image)

        self.cap.release()
        cv2.destroyAllWindows()
    


game = Game()
# game.splash_screen()
# game.winner()
# game.loser()
game.run_game(0)