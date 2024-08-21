import argparse
from file.evaluator import evaluation_model
from utils import LoadDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from module.ddpm import DDPM, CondDDPM
import torch
from diffusers import DDPMScheduler
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt

class Tester():

    def __init__(self, args):

        test_dataset = LoadDataset(data_path= "",
                                json_path= args.test_json_path,
                                objects_path= args.objects_file_path,
                                mode= "test")

        self.test_loader = DataLoader(test_dataset,
                                    batch_size= args.batch_size,
                                    num_workers= args.num_workers,
                                    pin_memory= True,
                                    shuffle= False)
        
        new_test_dataset = LoadDataset(data_path= "",
                                    json_path= args.test_json_path,
                                    objects_path= args.objects_file_path,
                                    mode= "new_test")

        self.new_test_loader = DataLoader(new_test_dataset,
                                    batch_size= args.batch_size,
                                    num_workers= args.num_workers,
                                    pin_memory= True,
                                    shuffle= False)

        self.device = args.device
        # model for training 
        self.train_model = DDPM().to(self.device)
        checkpoint = torch.load(args.pre_train, map_location=self.device)
        self.train_model.load_state_dict(checkpoint['model_state_dict'])
        # model for evaluation
        self.eval_model = evaluation_model(self.device)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps= 1000)

        self.de_normalize = transforms.Normalize(mean= [-1.0, -1.0, -1.0], std= [2.0, 2.0, 2.0])

    def inference(self):

        self.train_model.eval()
        with torch.no_grad():
            torch.manual_seed(3014)
            pbar = tqdm(enumerate(self.test_loader))
            for i, label in pbar:
                label = label.to(self.device)
                img = torch.randn(label.shape[0], 3, 64, 64).to(self.device)
                for t in self.noise_scheduler.timesteps:
                    residual = self.train_model(img, t, label)
                    img = self.noise_scheduler.step(residual, t, img).prev_sample
                    pbar.set_description_str(f"[Test] T: {t}")
                acc = self.eval_model.eval(img, label)
                de_img = self.de_normalize(img)
                self.pltImageGrid(de_img, "./result/test grid.png")
                print("\ntest acc ", acc)


        with torch.no_grad():
            pbar = tqdm(enumerate(self.new_test_loader))
            for i, label in pbar:
                label = label.to(self.device)
                img = torch.randn(label.shape[0], 3, 64, 64).to(self.device)
                for t in self.noise_scheduler.timesteps:
                    residual = self.train_model(img, t, label)
                    img = self.noise_scheduler.step(residual, t, img).prev_sample
                    pbar.set_description_str(f"[Test] T: {t}")
                    
                acc = self.eval_model.eval(img, label)     
                de_img = self.de_normalize(img)
                self.pltImageGrid(de_img, "./result/new test grid.png")
                print("\nnew test acc ", acc)
    
    def pltDenoiseProcess(self, images, path):
        plt.imshow(torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(images, nrow=10)))
        plt.savefig(path)

    def pltImageGrid(self, images, path):
        plt.imshow(torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(images, nrow=8)))
        plt.savefig(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type= str, default= "cuda:1", help= "training device")
    parser.add_argument("--test_json_path", type= str, default= "./file/new_test.json", help= "testing label path")
    parser.add_argument("--objects_file_path", type= str, default= "./file/objects.json", help= "objects json path which has all classification")
    parser.add_argument("--batch_size", type= str, default= 32 , help= "train batch size")
    parser.add_argument("--num_workers", type= int, default= 4, help= "number of worker")
    parser.add_argument("--pre-train", type= str, default= "./result/best.ckpt", help= "pre train model")

    args = parser.parse_args()

    tester = Tester(args)
    tester.inference()