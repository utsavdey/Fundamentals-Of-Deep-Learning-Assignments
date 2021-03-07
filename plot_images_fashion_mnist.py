"""Download the fashion-MNIST dataset and plot 1 sample image for each class."""

from keras.datasets import fashion_mnist
import wandb
# from wandb.keras import WandbCallback

wandb.init(project="MNIST_report",id="Question_1")
class_names = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
			   'Ankle boot']

(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
trainX=trainX / 255.0
testX=testX / 255.0

def log_images():
	# api = wandb.Api()
	# run = api.run("arup-das-cs20s016/Projects/plot_images/trails_123")
	set_images=[]
	set_labels=[]
	count=0
	for d in range(len(trainy)):
		if trainy[d]==count:
				set_images.append(trainX[d])
				set_labels.append(class_names[trainy[d]])
				count=count+1
		else:
				pass
		if count==10:
			break

	wandb.log({"Plot": [wandb.Image(img, caption=caption) for img, caption in zip(set_images, set_labels)]})
log_images()