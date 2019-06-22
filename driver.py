from nn_1 import *
from visualize import *

pygame.init()
screen = pygame.display.set_mode((700, 600))
screen.fill(BLACK)


if __name__ == '__main__':

	train_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	train_label = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
	test_X = []
	test_label = []
	model = train(train_X, train_label, epochs = 1000)
	# predict([0.99, 0.1], model) 
	max_val = calc_max_val(model)

	while not done:
	    for event in pygame.event.get():
	        if event.type == pygame.QUIT:
	            done = True
	    viz = Network_Viz(model, screen, max_val)
	    viz.create_network(100, [2,5,2])

	    if event.type == pygame.KEYDOWN:
	        if event.key == pygame.K_LEFT:
	            background_color = WHITE
	            print('left pressed')
	            screen.fill(background_color) #<--- Here
	            pygame.display.update()
	        screen.fill(BLACK)
	    pygame.display.flip()
