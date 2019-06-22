import argparse
from nn_1 import *
from visualize import *


pygame.init()
screen = pygame.display.set_mode((1000, 800))
screen.fill(BLACK)


if __name__ == '__main__':

	parser = argparse.ArgumentParser("Test")
	parser.add_argument("-p", help="Predict a value")
	parser.add_argument("-v", help="Visualize the network")

	args = parser.parse_args()
	train_X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0] ,[1, 0, 1], [1, 1, 0], [1,1, 1]])
	train_label = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [1,0], [1,0], [1,0], [1,0]])
	test_X = []
	test_label = []
	model = train(train_X, train_label, epochs = 1000)
	if args.p:
		predict([0,1,1], model) 

	if args.v:
		max_val = calc_max_val(model)

		while not done:
		    for event in pygame.event.get():
		        if event.type == pygame.QUIT:
		            done = True
		    viz = Network_Viz(model, screen, max_val)
		    viz.create_network(150, 75, [3,7,2])

		    if event.type == pygame.KEYDOWN:
		        if event.key == pygame.K_LEFT:
		            background_color = WHITE
		            print('left pressed')
		            screen.fill(background_color) #<--- Here
		            pygame.display.update()
		        screen.fill(BLACK)
		    pygame.display.flip()
