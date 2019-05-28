'''
Need to get coords once
Create links between layers
Update link color using weights

Future:
Update weights real time
'''


import pygame
pygame.init()
screen = pygame.display.set_mode((700, 600))
screen.fill([0, 0, 0])
done = False
NIGHT_GRAY = (104, 98, 115)
ORANGE = (255, 125, 0)


def create_circle(val, x, y):
    pygame.draw.circle(screen, (val, val, 0), (x, y), 15)


def create_link(val, startX, startY, endX, endY):
    pygame.draw.line(screen, val, [startX, startY], [endX, endY], 3)


def create_layer(x_coords, n_nodes):
    for i in range(n_nodes):
        create_circle(255, x_coords, 100 + 100 * i)


while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    for i in range(255):
        create_link(i, 30, 100, 90, 100)
        create_link(i, 30, 100, 90, 200)

    l1 = create_layer(30, 3)
    l2 = create_layer(90, 4)

    pygame.display.flip()
