import argparse
import torch
import torch.nn as nn
from network.Math_Module import P, Q
from network.decom import Decom
import os
import torchvision.transforms as transforms
from PIL import Image
import time
from utils import *

def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)

class Inference(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # loading decomposition model 
        self.model_Decom_low = Decom().to(self.device)
        self.model_Decom_low = load_initialize(self.model_Decom_low, self.opts.Decom_model_low_path)  # Device handled in load_initialize
        # loading R, old_model_opts, and L model
        self.unfolding_opts, self.model_R, self.model_L = load_unfolding(self.opts.unfolding_model_path)
        # loading adjustment model
        self.adjust_model = load_adjustment(self.opts.adjust_model_path).to(self.device)
        self.P = P().to(self.device)
        self.Q = Q().to(self.device)
        transform = [
            transforms.ToTensor(),
        ]
        self.transform = transforms.Compose(transform)
        print(self.model_Decom_low)
        print(self.model_R)
        print(self.model_L)
        print(self.adjust_model)

    def unfolding(self, input_low_img):
        input_low_img = input_low_img.to(self.device)
        for t in range(self.unfolding_opts.round):      
            if t == 0: # initialize R0, L0
                P, Q = self.model_Decom_low(input_low_img)
            else: # update P and Q
                w_p = (self.unfolding_opts.gamma + self.unfolding_opts.Roffset * t)
                w_q = (self.unfolding_opts.lamda + self.unfolding_opts.Loffset * t)
                P = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q(I=input_low_img, P=P, L=L, lamda=w_q) 
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
        return R, L
    
    def illumination_adjust(self, L, ratio):
        ratio = torch.ones(L.shape).to(self.device) * self.opts.ratio
        return self.adjust_model(l=L, alpha=ratio)
    
    def forward(self, input_low_img):
        with torch.no_grad():
            start = time.time()  
            R, L = self.unfolding(input_low_img)
            High_L = self.illumination_adjust(L, self.opts.ratio)
            I_enhance = High_L * R
            p_time = (time.time() - start)
        return I_enhance, p_time

    def run(self, low_img_path):
        file_name = os.path.basename(self.opts.img_path)
        name = file_name.split('.')[0]
        low_img = self.transform(Image.open(low_img_path)).unsqueeze(0).to(self.device)
        enhance, p_time = self.forward(input_low_img=low_img)
        if not os.path.exists(self.opts.output):
            os.makedirs(self.opts.output)
        save_path = os.path.join(self.opts.output, file_name.replace(name, f"{name}_{self.opts.ratio}_URetinexNet"))
        np_save_TensorImg(enhance.cpu(), save_path)  
        print(f"================================= time for {file_name}: {p_time:.6f} seconds ============================")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configure')
    
    # Paths: Modify these to match the structure on Google Colab
    parser.add_argument('--img_path', type=str, default="/content/demo/input/image.png")  # Input image path
    parser.add_argument('--output', type=str, default="/content/demo/output")  # Output directory
    parser.add_argument('--ratio', type=int, default=5)  # Ratio for illumination adjustment
    parser.add_argument('--Decom_model_low_path', type=str, default="./ckpt/init_low.pth")  # Decomposition model
    parser.add_argument('--unfolding_model_path', type=str, default="./ckpt/unfolding.pth")  # Unfolding model
    parser.add_argument('--adjust_model_path', type=str, default="./ckpt/L_adjust.pth")  # Adjustment model
    parser.add_argument('--gpu_id', type=int, default=0)
    
    opts = parser.parse_args()
    
    # Print all arguments
    for k, v in vars(opts).items():
        print(f"{k}: {v}")
    
    # Set GPU for PyTorch if available
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)
    
    # Run the inference
    model = Inference(opts)
    model.run(opts.img_path)
