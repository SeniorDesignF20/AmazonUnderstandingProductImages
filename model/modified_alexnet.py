import torch
import torch.nn as nn

class Modified_AlexNet(nn.Module):
	# Binary Classifier AlexNet that takes in 6 channel input

	def __init__(self, num_classes: int = 2, batch_size=64) -> None:
		super(Modified_AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(6, batch_size, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(batch_size, 3*batch_size, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(3*batch_size, 6*batch_size, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(6*batch_size, 4*batch_size, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(4*batch_size, 4*batch_size, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Sequential(
		    nn.Dropout(),
		    nn.Linear(4*batch_size * 6 * 6, batch_size*batch_size),
		    nn.ReLU(inplace=True),
		    nn.Dropout(),
		    nn.Linear(batch_size*batch_size, batch_size*batch_size),
		    nn.ReLU(inplace=True),
		    nn.Linear(batch_size*batch_size, batch_size),
		    nn.ReLU(inplace=True),
		    nn.Dropout(),
		    nn.Linear(batch_size, num_classes)
        )

		self.apply(self.init_weights)

	def init_weights(self, layer):
		if type(layer) == nn.Linear:
			torch.nn.init.xavier_uniform_(layer.weight)
			layer.bias.data.fill_(0.01)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x