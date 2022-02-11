import pygame
import random
import time
import neat
import os

WIN_WIDTH = 800
WIN_HEIGHT = 500
FLOOR_IMG = pygame.image.load(os.path.join("imgs", "floor.png"))
FLOOR_HEIGHT = WIN_HEIGHT/2 + 50
OBSTACLE = pygame.image.load(os.path.join("imgs", "obstacles.png"))

class Player:

    def __init__(self):
        self.x = 100
        self.y = WIN_HEIGHT/2
        self.vel_y = 0
        self.tick_count = 0
        self.height = self.y
        self.img = pygame.image.load(os.path.join("imgs", "player.png"))

    def jump(self):
        if self.y == WIN_HEIGHT/2:
            self.vel_y = -20


    def draw(self, win):
        win.blit(self.img, (self.x,self.y))

    def move(self):
        self.y += self.vel_y
        self.vel_y += 1.25

        if self.y > WIN_HEIGHT/2:
            self.y = WIN_HEIGHT/2

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Obstacle:
    GAP = 200
    VEL = 10
    
    def __init__(self, x):
        self.x = x
        self.img = OBSTACLE
        self.height = random.choice([50, 100, 150])

    def draw(self, win):
        win.blit(self.img, (self.x,FLOOR_HEIGHT-self.height))

    def move(self):
        if self.x < -50:
            self.height = random.choice([50, 100, 150])
            self.x = WIN_WIDTH + 50

        self.x -= self.VEL

    def collide(self, player):
        player_mask = player.get_mask()
        obstacle_mask = pygame.mask.from_surface(self.img)

        offset = (self.x-player.x, (FLOOR_HEIGHT - self.height) - round(player.y))

        point = player_mask.overlap(obstacle_mask, offset)

        if point:
            return True
        return False 


def draw_window(win, players, obstacle):
    obstacle.draw(win)

    win.blit(FLOOR_IMG, (0,FLOOR_HEIGHT))

    for player in players:
        player.draw(win)

    pygame.display.update()


def main(genomes, config):
    nets = []
    ge = []
    players = []
    run = True

    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    obstacle = Obstacle(500)

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        players.append(Player())
        g.fitness = 0
        ge.append(g)

    while run:
        if len(players) <= 0:
            run=False

        win.fill((255,255,255))
        clock.tick(80)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                run = False

        obstacle.move()
        for x, player in enumerate(players):
            player.move()
            ge[x].fitness += 2

            output = nets[x].activate((player.y, abs(obstacle.x - player.x)))
    
            if output[0] > 0.5:
                player.jump()
     
        for x, player in enumerate(players):

            if obstacle.collide(player):
                ge[x].fitness -= 5
                players.pop(x)
                nets.pop(x)
                ge.pop(x)        

        draw_window(win,players,obstacle)


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(main,50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)