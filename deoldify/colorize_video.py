import os
import argparse

import cv2
import torch
import torchvision.transforms as T

from model import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, required=True, help='path to the input video')
parser.add_argument('--output_path', type=str, required=True, help='path to save the output video')
parser.add_argument('--model_path', type=str, default='models/ColorizeVideo_gen.pth', help='path to the pretrained model')
parser.add_argument('--render_factor', type=int, default=21, help='render factor for the model, range 7-40')
args = parser.parse_args()

if not 7 <= args.render_factor <= 40:
    raise ValueError("Render factor should be in the range 7-40")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = Generator().to(device)
state_dict = torch.load(args.model_path)
model.load_state_dict(state_dict)
model.eval()

# Define image pre-processing
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Open input video
cap = cv2.VideoCapture(args.video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

with torch.no_grad():
    # Process each frame of the video
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to tensor
        img = transform(frame).unsqueeze(0).to(device)

        # Apply colorization
        stylized_img = model(img, args.render_factor)

        # Convert tensor back to image
        stylized_img = (stylized_img.squeeze(0).cpu().detach().clamp_(0, 1).numpy() * 255).astype('uint8')
        stylized_img = cv2.cvtColor(stylized_img, cv2.COLOR_RGB2BGR)

        # Write frame to output video
        out.write(stylized_img)

        # Print progress
        print(f'Processed frame {i + 1}/{total_frames}')

        # Release memory
        del img
        del stylized_img

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print('Done!')
