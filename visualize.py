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

def create_network(sp=100, n_nodes=[1]):
    '''
    INPUTS:
    sp: Spacing between neurons
    n_nodes: #nodes by layer
    
    '''

    x = 200
    for i in range(len(n_nodes)):
        if i == len(n_nodes)-1:
            create_layer(x, n_nodes[i], 0, x+2*sp)
        else:
            create_layer(x, n_nodes[i], n_nodes[i+1], x+2*sp)
        x += 2*sp


def create_circle(val, x, y):
    '''
    INPUTS:
    val: Color
    x, y: Position

    '''
    pygame.draw.circle(screen, (val, val, 0), (x, y), 15)


def create_link(val, startX, startY, endX, endY):
    '''
    INPUTS:
    val: Color
    startX, startY: Initial node position
    endX, endY: Final node position

    '''
    pygame.draw.line(screen, val, [startX, startY], [endX, endY], 3)


def create_layer(x_coords, n_nodes_i, n_nodes_next, x_coords_next):
    



    if n_nodes_next == 0:
        for i in range(n_nodes_i):
            create_circle(255, x_coords, 100 + 100 * i)
        return
    for i in range(n_nodes_i):
        t = 0
        for link in range(n_nodes_next):
            create_link(NIGHT_GRAY, x_coords, 100 + 100 * i, x_coords_next, (100 + 100 * (t)))
            t+=1
        create_circle(255, x_coords, 100 + 100 * i)
        # Create links
 




if __name__ == '__main__':    

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        create_network(100, [5,5,5])

        pygame.display.flip()
