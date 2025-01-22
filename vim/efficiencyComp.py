from timm.models import create_model
import torch
import models_mamba
import fvcore
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from prettytable import PrettyTable
from torchvision import transforms, datasets


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Creating model:")
model = create_model(
    'vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2',
    pretrained=False,
    num_classes=2,
    img_size=224,
)


checkpoint = torch.load('/home/T2410196/VisionMamba/vim/output/CANCER/vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2/best_checkpoint.pth', map_location='cpu')

checkpoint_model = checkpoint['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# interpolate position embedding
pos_embed_checkpoint = checkpoint_model['pos_embed']
embedding_size = pos_embed_checkpoint.shape[-1]
num_patches = model.patch_embed.num_patches
num_extra_tokens = model.pos_embed.shape[-2] - num_patches
# height (== width) for the checkpoint position embedding
orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
# height (== width) for the new position embedding
new_size = int(num_patches ** 0.5)
# class_token and dist_token are kept unchanged
extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
# only the position tokens are interpolated
pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
pos_tokens = torch.nn.functional.interpolate(
    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
checkpoint_model['pos_embed'] = new_pos_embed

model.load_state_dict(checkpoint_model, strict=False)

model.to(device)
print("Model loading success")

# Loading image
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
test_dataset = datasets.ImageFolder(root='/home/T2410196/VisionMamba/CANCER/test', transform=test_transforms)



# Inference FLOP count
image, label = test_dataset[0]  # Use the first image from the dataset
image = image.unsqueeze(0).to(device)
input_tensor = image
batch_size = 1

# flops = FlopCountAnalysis(model, image)
# total_flops = flops.total() / 1e9  # Convert to GFLOPs
params_table = parameter_count_table(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Inference Time
start_time = time.time()
for _ in range(10):  # Run for 10 iterations
    with torch.no_grad():
        model(input_tensor)
end_time = time.time()
batch_inference_time = (end_time - start_time) / 10
per_image_inference_time = batch_inference_time / batch_size

# GPU Memory Usage
if device == 'cuda':
    gpu_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    gpu_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
else:
    gpu_allocated = gpu_reserved = 0.0

# Throughput (FPS)
throughput = batch_size / batch_inference_time

from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ["Metric", "Value"]
# table.add_row(["FLOPs (GFLOPs)", f"{total_flops:.2f}"])
table.add_row(["Total Parameters", f"{total_params:,}"])
table.add_row(["Trainable Parameters", f"{trainable_params:,}"])
table.add_row(["Inference Time (Batch)", f"{batch_inference_time:.4f} seconds"])
table.add_row(["Inference Time (Per Image)", f"{per_image_inference_time:.4f} seconds"])
table.add_row(["Throughput (FPS)", f"{throughput:.2f} images/second"])
table.add_row(["GPU Memory Allocated", f"{gpu_allocated:.2f} GB"])
table.add_row(["GPU Memory Reserved", f"{gpu_reserved:.2f} GB"])

print(table)