import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#dataset
class FlowersDataset(Dataset):
    def __init__ (self, root_dir, transforms):
        super().__init__()
        self.root_dir = root_dir
        self.fimages = []
        self.classes = [fname for fname in os.listdir(root_dir)]
        
        for classes in self.classes:
            for fimage in os.listdir(root_dir + '/' + classes):
                if fimage.endswith('jpg'):
                    self.fimages.append([fimage, classes])
        self.transforms = transforms
        
    def __len__ (self):
        return len(self.fimages)
    
    def __getitem__(self, idx):
        fimage = self.fimages[idx][0]
        species = self.fimages[idx][1]        
        fpath = os.path.join(self.root_dir, species, fimage)
        image = np.array(Image.open(fpath).convert('RGB'))
        label = self.classes.index(species)
        if self.transforms is not None:
            transformed = self.transforms(image=image, label=label)
            image, label = transformed['image'], transformed['label']
            
#         image = torch.FloatTensor(image)
#         label = torch.LongTensor([label])
        return image, label