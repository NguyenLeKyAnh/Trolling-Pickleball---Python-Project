import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import os
import pygame
import random
import sys
from tkinter import *

# Initiatization
pygame.init()

# Configuration
win= Tk()
SCREEN_WIDTH = win.winfo_screenwidth()
SCREEN_HEIGHT = win.winfo_screenheight()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Trolling Pickleball")

# Color
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (150, 150, 150)
GREEN = (0, 255, 0)
PURPLE = (255, 0, 255)
CYAN = (0, 255, 255)

# FPS
clock = pygame.time.Clock()
FPS = 120

# Font
font = pygame.font.SysFont(None, 100)

# Variable 
istrain = 0
usedmodel = RandomForestRegressor()
predrate = 20
batch_size = 60
trainmode = 1

menu_music = pygame.mixer.Sound("Menu.mp3")
game_music = pygame.mixer.Sound("Game.mp3")
hit_sound = pygame.mixer.Sound("Hit.wav")
lose_sound = pygame.mixer.Sound("Lose.wav")

exe_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

save_path = os.path.join(exe_dir, "pickleball_grinder_ver_official.pkl")

# Ball
class Ball:
    def __init__(self, x, y, radius, speed, color, Clip_chance, color_dir, period, count):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed
        self.color = color
        self.Clip_chance = Clip_chance
        self.color_dir = color_dir
        self.period = period
        self.count = count
    def bounce(self, direction):
        if(direction == "horizontal" and self.Clip_chance <= 10 and (self.x - SCREEN_WIDTH / 2) * self.speed[0] < 0):
            if(self.y < -self.radius):
                self.y = SCREEN_HEIGHT + self.radius
                self.Clip_chance = random.randint(1, 100)
            elif(self.y > SCREEN_HEIGHT + self.radius):
                self.y = -self.radius
                self.Clip_chance = random.randint(1, 100)
        elif (direction == "horizontal"):
            self.speed = (self.speed[0], -self.speed[1])
            self.Clip_chance = random.randint(1, 100)
    def move(self):
        if(self.color != WHITE and self.color != GREEN and self.color != YELLOW  and self.color != CYAN):
            if(self.color == BLACK or self.color == PURPLE):
                self.color_dir = -self.color_dir
            self.color = (self.color[0] + self.color_dir, self.color[1], self.color[2] + self.color_dir)
        if(self.color == YELLOW):
            self.Lag()
        if(self.color[0] < 0 or self.color[2] < 0):
            self.color = (0, 0, 0)
        if(self.color[0] > 255 or self.color[2] > 255):
            self.color = (255, 0, 255)

        if (self.x < SCREEN_WIDTH / 2 + SCREEN_WIDTH / 4 and self.x > SCREEN_WIDTH / 2 - SCREEN_WIDTH / 4):
            self.Teleport(5)

        self.Horizontal_Bounce(3)

        self.x = self.x + self.speed[0]
        self.y = self.y + self.speed[1]
        if ((self.y < self.radius and self.speed[1] < 0) or (self.y > SCREEN_HEIGHT - self.radius and self.speed[1] > 0)):
            self.bounce("horizontal")
        if ((self.x - self.radius <= SCREEN_WIDTH / 2 and self.x - self.radius > SCREEN_WIDTH / 2 - self.radius / 2 and self.speed[0] < 0) or (self.x + self.radius >= SCREEN_WIDTH / 2 and self.x + self.radius < SCREEN_WIDTH / 2 + self.radius / 2 and self.speed[0] > 0)):
            self.Random_Bounce(10)


    def wall_collision(self):
        global blue_score
        global red_score

        if (self.x + self.radius < 0 or
            self.x - self.radius > SCREEN_WIDTH):

            if(self.x - self.radius > SCREEN_WIDTH):
                LRModel.store_pending_output(ball.y)
                
                if((red_score + blue_score) % batch_size == (batch_size - 1) and istrain):
                    LRModel.train_model_with_result()
            else:
                LRModel.pending_inputs = []

            score = 1
            if(self.color == GREEN):
                score = 5
            if (self.x + self.radius < 0):
                red_score += score
            else:
                blue_score += score

            if (random.randint(1, 100) <= 5):
                self.color = GREEN
                self.radius = 20
            elif (random.randint(1, 100) <= 10):
                self.color = CYAN
                self.radius = 10
            elif (random.randint(1, 100) <= 15):
                self.color = YELLOW
                self.radius = 20
            elif (random.randint(1, 100) <= 20):
                self.color = PURPLE
                self.radius = 20
            else:
                self.color = WHITE
                self.radius = 20
            lose_sound.play()
            self.x = SCREEN_WIDTH // 2
            self.y = SCREEN_HEIGHT // 2
            self.speed = (2 * -self.speed[0] / abs(self.speed[0]), 2 - 4 * random.randint(0, 1))
            blue_basket.Clip_chance2 = random.randint(1, 1000)
            red_basket.Clip_chance2 = random.randint(1, 1000)

    def Random_Bounce(self, chance_bypercent):
        ran_int = random.randint(1, 100)
        if(ran_int <= chance_bypercent):
            self.speed = (-self.speed[0], self.speed[1])

    def Teleport(self, chance_byper10000_perframe):
        ran_int = random.randint(1, 10000)
        if(ran_int <= chance_byper10000_perframe):
            self.x = self.x + abs(self.speed[0]) / self.speed[0] * SCREEN_WIDTH / 6

    def Horizontal_Bounce(self, chance_byper10000_perframe):
        ran_int = random.randint(1, 10000)
        if(ran_int <= chance_byper10000_perframe):
            self.speed = (self.speed[0], -self.speed[1])

    def Lag(self):
        if (self.count >= self.period):
            self.x = self.x - self.speed[0]
            self.y = self.y - self.speed[1]

            self.count -= 1
            self.period = random.randint(12, 240)
        else:
            self.x = self.x + self.speed[0]
            self.y = self.y + self.speed[1]
            self.count += 1


ball = Ball(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, 20, (2 - 4 * random.randint(0, 1), 2 - 4 * random.randint(0, 1)), WHITE, random.randint(1, 100), -3, random.randint(12, 240), 0)
testball = Ball(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, 20, (2 - 4 * random.randint(0, 1), 2 - 4 * random.randint(0, 1)), RED, random.randint(1, 100), -3, random.randint(12, 240), 0)

# Boundary
class Boundary:
    def __init__(self, x, y, length, width, color):
        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.color = color

boundary = Boundary((SCREEN_WIDTH // 2) - 3, 0, SCREEN_HEIGHT, 6, WHITE)

# Speed random shifting
class Speed_random_shifting:
    def __init__(self, period, count):
        self.period = period
        self.count = count
    def ball_speed_random_change(self):
        self.count += 1/FPS
        if (self.count >= self.period):

            change_speed = random.randint(2, 10)
            ball_direction = (ball.speed[0] // (abs(ball.speed[0])), ball.speed[1] // (abs(ball.speed[1])))
            ball.speed = (change_speed * ball_direction[0], change_speed * ball_direction[1])

            self.count = 0
            self.period = random.randint(2, 5)
ball_speed_random_change = Speed_random_shifting(1, 0)

# Basket
class Basket:
    def __init__(self, name, x, y, length, width, speed, color, Clip_chance2):
        self.name = name
        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.speed = speed
        self.color = color
        self.Clip_chance2 = Clip_chance2
    def move(self):
        if(ball.x < SCREEN_WIDTH // 8 or ball.x > SCREEN_WIDTH // 8 * 7):
            self.Teleport(20, SCREEN_HEIGHT // 6)

        self.y = self.y + self.speed
        if (self.y < 0):
            self.speed = abs(self.speed)
        elif (self.y > SCREEN_HEIGHT - self.length):
            self.speed = -abs(self.speed)
    def change_direction(self):
        self.speed = -self.speed
    def ball_collision(self):
        if(self.Clip_chance2 <= 995):
            if (self.name == "blue_basket"):
                if (ball.x - ball.radius < self.x + self.width and
                    self.y - ball.radius < ball.y < self.y + self.length + ball.radius and
                    ball.x > self.x):
                    ball.speed = (abs(ball.speed[0]), ball.speed[1])
                    self.Clip_chance2 = random.randint(1, 1000)
                    hit_sound.play()
            elif (self.name == "red_basket"):
                if (ball.x + ball.radius > self.x and
                    self.y - ball.radius < ball.y < self.y + self.length + ball.radius and
                    ball.x < self.x):
                    LRModel.store_pending_output(ball.y)

                    ball.speed = (-abs(ball.speed[0]), ball.speed[1])
                    self.Clip_chance2 = random.randint(1, 1000)
                    hit_sound.play()
    def Teleport(self, chance_byper10000_perframe, tele_dist):
        if((ball.x < SCREEN_WIDTH // 8 and self.name == "blue_basket" and ball.speed[0] < 0) or (ball.x > (SCREEN_WIDTH // 8 * 7) and self.name == "red_basket" and ball.speed[0] > 0)):
            ran_int = random.randint(1, 10000)
            if(ran_int <= chance_byper10000_perframe):
                self.y = self.y + tele_dist - random.randint(0, 1) * 2 * tele_dist
            if(self.y < 0):
                self.y = 0
            if(self.y + self.length > SCREEN_HEIGHT):
                self.y = SCREEN_HEIGHT - self.length

blue_basket = Basket("blue_basket", 20, SCREEN_HEIGHT // 2, 150, 15, -3, BLUE, random.randint(1, 1000))
red_basket = Basket("red_basket", SCREEN_WIDTH - 15 - 20, SCREEN_HEIGHT // 2, 150, 15, -3, RED, random.randint(1, 1000))

#Basket_Change
class Size_random_shifting:
    def __init__(self, period, count, duration, length):
        self.period = period
        self.count = count
        self.duration = duration
        self.length = length
    def basket_size_random_change(self):
        self.count += 1/FPS
        if (self.count >= self.period + self.duration):
            blue_basket.length = self.length
            red_basket.length = self.length
            self.count = 0
            self.period = random.randint(20, 40)
            self.duration = random.randint(5, 10)
        elif (self.count >= self.period):
            blue_basket.length = self.length // 2
            red_basket.length = self.length // 2
basket_size_random_change = Size_random_shifting(random.randint(20, 40), 0, random.randint(5, 10), red_basket.length)

# Score
blue_score = 0
red_score = 0

def scoreboard():
    blue_score_text = f"{blue_score}"
    red_score_text = f"{red_score}"

    blue_score_text_size = font.size(blue_score_text)[0]
    red_score_text_size = font.size(red_score_text)[0]

    blue_score_text = font.render(blue_score_text, True, BLUE)
    red_score_text = font.render(red_score_text, True, RED)

    screen.blit(blue_score_text, ((SCREEN_WIDTH // 2) - blue_score_text_size - 30, 10))
    screen.blit(red_score_text, ((SCREEN_WIDTH // 2) + 30, 10))

FPScount = 0
# Draw_screen
def draw_screen(frame):
    screen.fill(BLACK)
    pygame.draw.circle(screen, ball.color, (ball.x, ball.y), ball.radius)
    #pygame.draw.circle(screen, RED, (testball.x, testball.y), testball.radius)
    pygame.draw.rect(screen, blue_basket.color, (blue_basket.x, blue_basket.y, blue_basket.width, blue_basket.length))
    pygame.draw.rect(screen, red_basket.color, (red_basket.x, red_basket.y, red_basket.width, red_basket.length))
    pygame.draw.rect(screen, boundary.color, (boundary.x, boundary.y, boundary.width, boundary.length))

    tut(frame)

    blue_basket.move()
    red_basket.move()
    ball.move()

    blue_basket.ball_collision()
    red_basket.ball_collision()
    ball.wall_collision()

    ball_speed_random_change.ball_speed_random_change()
    basket_size_random_change.basket_size_random_change()

    scoreboard()

    pygame.display.flip()

def predidcty(istored, evalpos, speed):
    if(istored == 1):
        remdist = (SCREEN_WIDTH - evalpos[0])
    else:
        remdist = evalpos[0]
    reremdist = remdist
    isflip = 0
    if(speed > 0):
        if(remdist - (SCREEN_HEIGHT - evalpos[1]) - SCREEN_HEIGHT > 0):
            reremdist = remdist - (SCREEN_HEIGHT - evalpos[1]) - SCREEN_HEIGHT
        elif(remdist - (SCREEN_HEIGHT - evalpos[1]) > 0):
            isflip = 1
            reremdist = remdist - (SCREEN_HEIGHT - evalpos[1])
            reremdist = SCREEN_HEIGHT - reremdist
        else:
            reremdist = evalpos[1] + remdist

    else:
        if(remdist - evalpos[1] - SCREEN_HEIGHT > 0):
            reremdist = remdist - evalpos[1] - SCREEN_HEIGHT
            reremdist = SCREEN_HEIGHT - reremdist
        elif(remdist - evalpos[1] > 0):
            isflip = 1
            reremdist = remdist - evalpos[1]
        else:
            reremdist = evalpos[1] - remdist

    posy = reremdist
    if(istored == 1 or istored == 2):
        return posy
    else:
        if(isflip == 1):
            speed *= -1
        return predidcty(1, (blue_basket.x, posy), speed)

class Model:
    # AI
    def __init__(self, can_train, X_train, y_train, model, pending_inputs, pending_train_data, trainint, testpos):
        self.can_train = can_train
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.pending_inputs = pending_inputs
        self.pending_train_data = pending_train_data
        self.trainint = trainint
        self.testpos = testpos

    # Prepare Training Data
    def store_pending_input(self, input_features):
        self.pending_inputs.append(input_features)

    # Prepare Training Results
    def store_pending_output(self, target_position):
        if not self.pending_inputs:
            return

        # Pair all pending inputs with their corresponding targets
        for input_features in self.pending_inputs:
            self.pending_train_data.append((input_features, target_position))

        # Clear pending inputs
        self.pending_inputs = []

    # Train AI
    def train_model_with_result(self):
        if self.can_train:
            print("Training")
            if not self.pending_train_data:
                return

            # Shuffle the data
            random.shuffle(self.pending_train_data)
            shuffled_features, shuffled_targets = zip(*self.pending_train_data)
            self.pending_train_data = []

            # Convert to NumPy arrays
            X_train_np = np.array(shuffled_features)
            y_train_np = np.array(shuffled_targets)

            # Retrain the model on the entire batch
            self.model.fit(X_train_np, y_train_np)

            # Calculate loss
            predictions = self.model.predict(X_train_np)
            loss = mean_squared_error(y_train_np, predictions)
            print(f"Loss: {loss:.5f}")

            # Save the updated model
            with open(save_path, 'wb') as f:
                pickle.dump(self.model, f)

LRModel = Model(False, [], 0, usedmodel, [], [], 0, 0)

# AI
def AI(diff):
    posy = 0
    if(diff == 1):
        posy = ball.y
    elif(diff == 3):
        if(ball.speed[0] > 0):
            posy = predidcty(1, (ball.x, ball.y), ball.speed[1])
        else:
            posy = ball.y

    elif(diff == 2):
        pred_y = 0
        x_pos = 0
        is_right_dir = 0
        clip_flag = 0
        can_tele_flag = 0
        can_bounce_flag = 0

        if(ball.speed[0] > 0):
            is_right_dir = 1
            pred_y = predidcty(1, (ball.x, ball.y), ball.speed[1])
            can_bounce_flag = (ball.x < SCREEN_WIDTH / 2)
        else:
            is_right_dir = 0
            pred_y = predidcty(0, (ball.x, ball.y), ball.speed[1])
            can_bounce_flag = (ball.x > SCREEN_WIDTH / 2)
        x_pos = ball.x
        clip_flag = (ball.Clip_chance <= 10 and (ball.x - SCREEN_HEIGHT) * ball.speed[0] < 0)
        can_tele_flag = (ball.x < SCREEN_WIDTH // 8 or ball.x > SCREEN_WIDTH // 8 * 7)

        new_input = [pred_y, x_pos, is_right_dir, clip_flag, can_tele_flag, can_bounce_flag]
        LRModel.store_pending_input(new_input)

        LRModel.trainint += 1
        if(LRModel.trainint == predrate):
            LRModel.trainint = 0
            try:
                posy = float(LRModel.model.predict(np.array([new_input]))[0])
                LRModel.testpos = posy
            except:
                posy = pred_y
        posy = LRModel.testpos
    elif(diff == 4):
        clip_flag = (ball.Clip_chance <= 10 and (ball.x - SCREEN_WIDTH / 2) * ball.speed[0] < 0)
        dist_y = 0
        can_tele_flag = (ball.x < SCREEN_WIDTH / 2 + SCREEN_WIDTH / 4 and ball.x > SCREEN_WIDTH / 2 - SCREEN_WIDTH / 4)
        if(ball.speed[1] > 0):
            dist_y = SCREEN_HEIGHT - ball.y
        else:
            dist_y = ball.y

        if(ball.speed[0] > 0):
            posy = predidcty(1, (ball.x, ball.y), ball.speed[1])

            if(clip_flag and ball.x + dist_y < SCREEN_WIDTH / 2):
                posy = SCREEN_HEIGHT - posy
            elif can_tele_flag:
                telepos = predidcty(1, (ball.x + SCREEN_WIDTH / 6, ball.y), ball.speed[1])
                posy = (posy + telepos) / 2

        else:
            if(ball.x > SCREEN_WIDTH / 2):
                posy = predidcty(2, (ball.x, ball.y), ball.speed[1])
            else:
                posy = predidcty(0, (ball.x, ball.y), ball.speed[1])
            
            if(clip_flag and ball.x - dist_y > SCREEN_WIDTH / 2):
                posy = SCREEN_HEIGHT - posy

    if(istrain):
        if(trainmode == 0):
            blue_basket.length = 2000
        if(trainmode == 1):
            posblue = predidcty(2, (ball.x, ball.y), ball.speed[1])
            if((posblue - (blue_basket.y + blue_basket.length // 2)) * blue_basket.speed < 0) and blue_basket.y + blue_basket.length < SCREEN_HEIGHT and blue_basket.y > 0:
                blue_basket.change_direction()
        if(trainmode == 2):
            posblue = ball.y
            if((posblue - (blue_basket.y + blue_basket.length // 2)) * blue_basket.speed < 0) and blue_basket.y + blue_basket.length < SCREEN_HEIGHT and blue_basket.y > 0:
                blue_basket.change_direction()
    if((posy - (red_basket.y + red_basket.length // 2)) * red_basket.speed < 0) and red_basket.y + red_basket.length < SCREEN_HEIGHT and red_basket.y > 0:
        red_basket.change_direction()
    testball.x = red_basket.x
    testball.y = LRModel.testpos


# Main Menu
def menu():
    screen.fill(BLACK)

    title = "TROLLING_PICKLEBALL"
    title_text = f"{title}"

    text_size = font.size(title_text)[0]
    title_text = font.render(title_text, True, WHITE)

    screen.blit(title_text, ((SCREEN_WIDTH // 2) - text_size // 2, 150))
    #Title

    title = "Press [SPACE] to continue"
    title_text = f"{title}"

    text_size = font.size(title_text)[0]
    title_text = font.render(title_text, True, WHITE)

    screen.blit(title_text, ((SCREEN_WIDTH // 2) - text_size // 2, 500))
    #Title

    pygame.display.flip()

# Game Over
def over():
    screen.fill(BLACK)

    over_text = "TIE!"
    text_col = WHITE
    if(red_score > blue_score):
        over_text = "RED WON!"
        text_col = RED
    if(blue_score > red_score):
        over_text = "BLUE WON!"
        text_col = BLUE

    over_string = f"{over_text}"

    text_size = font.size(over_string)[0]
    over_string = font.render(over_string, True, text_col)

    screen.blit(over_string, ((SCREEN_WIDTH // 2) - text_size // 2, 150))
    #Text

    over_string = f"{red_score}"

    text_size = font.size(over_string)[0]
    over_string = font.render(over_string, True, RED)
    screen.blit(over_string, ((SCREEN_WIDTH // 2) - text_size // 2 + 100, 500))

    over_string = f"{blue_score}"

    text_size = font.size(over_string)[0]
    over_string = font.render(over_string, True, BLUE)
    screen.blit(over_string, ((SCREEN_WIDTH // 2) - text_size // 2 - 100, 500))
    #Score
    
    pygame.display.flip()

# Game Mode
def gamemode():
    screen.fill(BLACK)
    over_string = "Choose the number of player"

    text_size = font.size(over_string)[0]
    over_string = font.render(over_string, True, WHITE)
    screen.blit(over_string, ((SCREEN_WIDTH // 2) - text_size // 2, 150))

    over_string = "1 or 2"

    text_size = font.size(over_string)[0]
    over_string = font.render(over_string, True, WHITE)
    screen.blit(over_string, ((SCREEN_WIDTH // 2) - text_size // 2, 500))

    pygame.display.flip()

# Diffliculty
def choosediff():
    screen.fill(BLACK)
    over_string = "Choose difficulty"

    text_size = font.size(over_string)[0]
    over_string = font.render(over_string, True, WHITE)
    screen.blit(over_string, ((SCREEN_WIDTH // 2) - text_size // 2, 150))

    over_string = "1: Easy"

    text_size = font.size(over_string)[0]
    over_string = font.render(over_string, True, WHITE)
    screen.blit(over_string, ((SCREEN_WIDTH // 2) - text_size // 2, 300))

    over_string = "2: Medium"

    text_size = font.size(over_string)[0]
    over_string = font.render(over_string, True, WHITE)
    screen.blit(over_string, ((SCREEN_WIDTH // 2) - text_size // 2, 400))

    over_string = "3: Hard"

    text_size = font.size(over_string)[0]
    over_string = font.render(over_string, True, WHITE)
    screen.blit(over_string, ((SCREEN_WIDTH // 2) - text_size // 2, 500))

    over_string = "4: Impossible"

    text_size = font.size(over_string)[0]
    over_string = font.render(over_string, True, WHITE)
    screen.blit(over_string, ((SCREEN_WIDTH // 2) - text_size // 2, 600))

    pygame.display.flip()

# Tutorial
def tut(frame):
    if(frame < 3):

        over_string = "Press [Q] and [P]"

        text_size = font.size(over_string)[0]
        over_string = font.render(over_string, True, WHITE)
        screen.blit(over_string, ((SCREEN_WIDTH // 2) - text_size // 2, 600))

# Main
def main():
    print(save_path)
    if istrain:
        FPS = 960
        predrate = FPS / 6
    else:
        FPS = 120
        predrate = FPS / 6

    try:
        with open(save_path, 'rb') as f:
            LRModel.model = pickle.load(f)
    except FileNotFoundError:
        LRModel.model = usedmodel
        print("Restart")

    running = True

    stages = 0
    playernum = 0
    FPScount = 0
    diff = 0

    menu_music.play(-1)
    while running:

        FPScount += 1 / FPS
        if(red_score + blue_score >= 100 and not istrain):
            stages = 2
        # Keypress
        if(stages == 1):
            if(playernum == 1):
                AI(diff)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_q:
                        blue_basket.change_direction()
                    if event.key == pygame.K_p and playernum == 2:
                        red_basket.change_direction()
            # Drawing
            draw_screen(FPScount)
            clock.tick(FPS)
        elif(stages == 0):
            menu()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        stages = 3
                        gamemode()
        elif(stages == 3):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        stages = 4
                        playernum = 1
                        choosediff()
                    elif event.key== pygame.K_2:
                        FPScount = 0
                        stages = 1
                        playernum = 2

                        menu_music.stop()
                        game_music.play(-1)
        elif(stages == 4):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        diff = 1
                        FPScount = 0
                        stages = 1

                        menu_music.stop()
                        game_music.play(-1)
                    elif event.key == pygame.K_2:
                        trainint = 0
                        diff = 2
                        LRModel.can_train = istrain
                        FPScount = 0
                        stages = 1

                        menu_music.stop()
                        game_music.play(-1)
                    elif event.key == pygame.K_3:
                        diff = 3
                        FPScount = 0
                        stages = 1

                        menu_music.stop()
                        game_music.play(-1)
                    elif event.key == pygame.K_4:
                        diff = 4
                        FPScount = 0
                        stages = 1

                        menu_music.stop()
                        game_music.play(-1)
        else:
            over()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

    pygame.quit()
    sys.exit()

# Run
if __name__ == "__main__":
    main()


#Trolls:
#1/ Speed change
#2/ Vertical bounce
#3/ Teleportation
#4/ Paddle size change
#5/ Loop
#6/ Horizontal bounce
#7/ Clip
#8/ Green
#9/ Purple
#10/ Paddle Teleport
#11/ Yellow
#12/ Cyan