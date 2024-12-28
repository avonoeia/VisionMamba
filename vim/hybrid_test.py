import torch
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
import torch.nn.functional as F
import argparse
import numpy as np
from timm.models import create_model
import models_mamba

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    test_dataset = datasets.ImageFolder(args.test_data_path, transform=test_transforms)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True, # Will drop last few images if it doesn't fit batch size.
    )

    args.nb_classes = len(test_dataset.classes)
    args.input_size = 224

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        img_size=args.input_size
    )

    
    checkpoint = torch.load(args.model_checkpoint_path, map_location='cpu')

    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['mamba.pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.mamba.patch_embed.num_patches
    num_extra_tokens = model.mamba.pos_embed.shape[-2] - num_patches
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
    checkpoint_model['mamba.pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    evaluate_model_with_metrics(model, test_data_loader, device, binary_classification=True)



def get_args_parser():
    parser = argparse.ArgumentParser('Vim test script', add_help=False)

    # Model arguments
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to test')
    parser.add_argument('--model-checkpoint-path', type=str, 
                        help="Enter absolute path to your trained (or finetuned) model checkpoint. Checkpoint files end with .pth")
    
    # Data arguments
    parser.add_argument('--test-data-path', default='./data/test', type=str, help="Enter the absoloute path to your data")
    parser.add_argument('--is-binary', default=False, type=bool)
    
    # Params
    parser.add_argument('--batch-size', type=int, help="Enter batch size")
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--num_workers', default=10, type=int)
    

    return parser

def evaluate_model_with_metrics(model, loader, device, binary_classification=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get model outputs
            outputs = model(images)
            
            # For DeiT models, extract logits
            if hasattr(outputs, 'logits'):  
                outputs = outputs.logits
            
            # Get predicted classes
            _, predicted = outputs.max(1)

            # Accumulate metrics
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Probabilities for ROC AUC
            if binary_classification:
                probabilities = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probabilities)

    # Accuracy
    accuracy = 100. * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(all_labels),
                yticklabels=np.unique(all_labels), cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show(block=False)
    plt.pause(0.001)

    # Precision, Recall, F1-Score
    precision = precision_score(all_labels, all_preds, average='binary' if binary_classification else 'weighted')
    recall = recall_score(all_labels, all_preds, average='binary' if binary_classification else 'weighted')
    f1 = f1_score(all_labels, all_preds, average='binary' if binary_classification else 'weighted')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # AUC-ROC and ROC Curve (for binary classification)
    if binary_classification:
        all_probs = np.array(all_probs)
        if len(np.unique(all_labels)) == 2: 
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            auc_score = auc(fpr, tpr)
            print(f"AUC-ROC: {auc_score:.4f}")

            # Plot ROC Curve
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid()
            plt.show(block=False)
            plt.pause(0.001)
        else:
            print("AUC-ROC is not supported for multi-class tasks without modification.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Vim test script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)