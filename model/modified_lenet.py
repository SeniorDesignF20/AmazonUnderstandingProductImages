import torch
import torch.nn as nn

class Modified_LeNet(nn.Module):

	def __init__(self, num_classes=2, batch_size=64):
		super(Modified_LeNet, self).__init__()

		num_channels = 6

		self.features = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=6*num_channels, kernel_size=5, stride=1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=6*num_channels, out_channels=16*num_channels, kernel_size=5, stride=1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=16*num_channels, out_channels=120*num_channels, kernel_size=5, stride=1),
			nn.Tanh()
		)

		self.classifier = nn.Sequential(
			nn.Linear(in_features=120*num_channels, out_features=84*num_channels),
			nn.Tanh(),
			nn.Linear(in_features=84*num_channels, out_features=num_classes)
		)

	def init_weights(self, layer):
		if type(layer) == nn.Linear:
			torch.nn.init.xavier_uniform_(layer.weight)
			layer.bias.data.fill_(0.01)

	def forward(self, x):
		x = self.features()
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x

