'''
Need to get coords once
Create links between layers
Update link color using weights

Future:
Update weights real time
'''


import pygame
import numpy as np

done = False
NIGHT_GRAY = (104, 98, 115)
ORANGE = (255, 125, 0)
WHITE = (255,255,255)
BLACK = (0, 0, 0)


class Network_Viz:

    def __init__(self, model,screen=None, max_val = 0):
        pygame.init()
        self.model = model
        self.max_val = max_val

        if screen == None:
            self.screen = pygame.display.set_mode((700, 600))
            self.screen.fill(BLACK)
        else:
            self.screen = screen

    def get_screen(self):
        return self.screen

    def create_network(self, sp=100, n_nodes=[1]):
        '''
        INPUTS:
        sp: Spacing between neurons
        n_nodes: #nodes by layer
        
        '''

        x = 200
        for i in range(len(n_nodes)):
            if i == len(n_nodes)-1:
                self.create_layer(x, n_nodes[i], 0, x+2*sp, i)
            else:
                self.create_layer(x, n_nodes[i], n_nodes[i+1], x+2*sp, i)
            x += 2*sp
            

    def create_circle(self, val, x, y):
        '''
        INPUTS:
        val: Color
        x, y: Position

        '''
        pygame.draw.circle(self.screen, (val, val, 0), (x, y), 15)


    def create_link(self, val, startX, startY, endX, endY):
        '''
        INPUTS:
        val: Color
        startX, startY: Initial node position
        endX, endY: Final node position

        '''
        pygame.draw.line(self.screen, val, [startX, startY], [endX, endY], 3)



    def create_layer(self, x_coords, n_nodes_i, n_nodes_next, x_coords_next, Wi):

        factor = 100

        if n_nodes_next == 0:
            for i in range(n_nodes_i):
                self.create_circle(255, x_coords, 100 + 100 * i)
            return

        for i in range(n_nodes_i):
            t = 0
            for link in range(n_nodes_next):
                val = self.model[Wi][i][link]
                scaled_val = val / self.max_val

                if scaled_val > 0:
                    color = (0, scaled_val*factor, 0)
                else:
                    color = (np.abs(scaled_val)*factor, 0, 0)

                self.create_link(color, x_coords, 100 + 100 * i, x_coords_next, (100 + 100 * (t)))
                t+=1
            self.create_circle(255, x_coords, 100 + 100 * i)
            # Create links
     




if __name__ == '__main__':    
    pygame.init()
    screen = pygame.display.set_mode((700, 600))
    screen.fill(BLACK)

    c_Test= Network_Viz()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        c_Test.create_network(100, [2,5,2])
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                background_color = WHITE
                print('left pressed')
                screen.fill(background_color) #<--- Here
                pygame.display.update()
            screen.fill(BLACK)

        pygame.display.flip()
