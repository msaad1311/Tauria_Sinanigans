import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from Models.cartoon_models.generator import Generator
from Models.segment_models.models.modnet import MODNet

import numpy as np
import gc
import cv2
import matplotlib.pyplot as plt
import os

checkpoint_seg = r'Checkpoints\Segmentation.ckpt'
checkpoint_car = r'Checkpoints\Cartoonization.pth'

device = 'cuda' if torch.cuda.is_available() else  'cpu'

car_generator = Generator().to(device)
car_generator.load_state_dict(torch.load(checkpoint_car))

seg_model = MODNet(backbone_pretrained=False)
seg_model = nn.DataParallel(seg_model)
seg_model = seg_model.to(device)
seg_model.load_state_dict(torch.load(checkpoint_seg))

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

print('Init WebCam...')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print('Start matting...')
while(True):
    _, frame_np = cap.read()
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
    frame_np = frame_np[:, 120:792, :]
    frame_np = cv2.flip(frame_np, 1)
    tensor_img = preprocess(frame_np).unsqueeze(0).to(device)
    seg_model.eval()
    with torch.no_grad():
        _, _, matte_tensor = seg_model(tensor_img, True)

    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
    
    background = cv2.imread('backgrounds/background3.jpg')
    background = cv2.resize(background,(matte_np.shape[1],matte_np.shape[0]))
    background = cv2.cvtColor(background,cv2.COLOR_BGR2RGB)
    
    fg_np = np.array(matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, background))
    fg_np = fg_np.astype(np.uint8)
    fg_np = cv2.cvtColor(fg_np,cv2.COLOR_BGR2RGB)
#     fg_np = torch.tensor(fg_np)
    fg_np1 = preprocess(fg_np).unsqueeze(0).to(device).to(torch.float32)
    car_generator.eval()
    
    with torch.no_grad():
        cartoonImage = car_generator(fg_np1)
    car_generator.train()
    cartoonImage1 = inv_normalize(cartoonImage).squeeze(0).permute(1,2,0).cpu().numpy()
    cartoonImage2 = cv2.cvtColor(cartoonImage1,cv2.COLOR_RGB2BGR)

    cv2.imshow('MODNet - WebCam [Press \'Q\' To Exit]', cartoonImage1)#cartoonImage.squeeze(0).permute(1,2,0).cpu().numpy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
print('Exit...')