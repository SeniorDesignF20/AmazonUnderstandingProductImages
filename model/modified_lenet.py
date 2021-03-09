import torch
import torch.nn as nn

class Modified_LeNet(nn.Module):

	def __init__(self, num_classes=2, batch_size=64, dim=7):
		super(Modified_LeNet, self).__init__()

		# Assume the input is 64 x 6 x H x W, H=W (I dont know what happens when H != W). 
		# Then dim should be floor((floor(H/2) - 2)/2) - 6
		# So, for example 56x56 image: dim = floor((floor(56/2) - 2)/2) - 6 -> floor(13) - 6 -> 7

		num_channels = 6
		bs = batch_size
		bs1 = int(bs)

		self.features = nn.Sequential(
			nn.Conv2d(in_channels=num_channels, out_channels=bs, kernel_size=5, stride=1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=bs, out_channels=bs1, kernel_size=5, stride=1),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2),
			nn.Conv2d(in_channels=bs1, out_channels=bs1, kernel_size=5, stride=1),
			nn.Tanh()
		)

		self.classifier = nn.Sequential(
			nn.Linear(in_features=bs1 * dim * dim, out_features=bs1),
			nn.Tanh(),
			nn.Linear(in_features=bs1, out_features=num_classes)
		)

	def init_weights(self, layer):
		if type(layer) == nn.Linear:
			torch.nn.init.xavier_uniform_(layer.weight)
			layer.bias.data.fill_(0.01)

	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x