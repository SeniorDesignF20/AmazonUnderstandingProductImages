from moveup import moveup
from deleteEmptyFolders import deleteEmptyFolders
from bundle import bundle
import os

path = os.path.join(os.getcwd(), 'Bags')

moveup(path)
deleteEmptyFolders(path)
bundle(path)