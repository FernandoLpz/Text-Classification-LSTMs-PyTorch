from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyMapDataset(Dataset):
	def __init__(self, data):
		self.data = data
		self.target = [0,0,1,1,0,0,1,1,0,0,1,1]
		
	def __len__(self):
		return len(self.data)
		
	def __getitem__(self, idx):
	
	 return self.data[idx], self.target[idx]
	 
data = [0,1,2,3,4,5,6,7,8,9,10,11]

map_dataset = MyMapDataset(data)

loader = DataLoader(map_dataset, batch_size=2)

for x,y in loader:
	print(x,'-',y)