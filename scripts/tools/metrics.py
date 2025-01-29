import torch
import torchmetrics
from tqdm import tqdm
import matplotlib.pyplot as plt

def test(model, test_loader, device, iou_threshold=0.5):
    """Test the model on the test set and calculate metrics."""
    model.eval()
    
    # Initialize metrics
    precision_metric = torchmetrics.Precision(num_classes=2, average='macro', compute_on_step=False).to(device)
    recall_metric = torchmetrics.Recall(num_classes=2, average='macro', compute_on_step=False).to(device)
    f1_metric = torchmetrics.F1(num_classes=2, average='macro', compute_on_step=False).to(device)
    iou_metric = torchmetrics.MeanAveragePrecision(iou_thresholds=[iou_threshold], compute_on_step=False).to(device)
    
    # Tracking lists
    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Make predictions
            predictions = model(images)

            # Extract predicted boxes and labels
            pred_boxes = [output['boxes'] for output in predictions]
            pred_labels = [output['labels'] for output in predictions]

            # Extract ground truth boxes and labels
            true_boxes = [target['boxes'] for target in targets]
            true_labels = [target['labels'] for target in targets]
            
            # Update metrics
            precision_metric.update(pred_labels, true_labels)
            recall_metric.update(pred_labels, true_labels)
            f1_metric.update(pred_labels, true_labels)
            iou_metric.update(pred_boxes, true_boxes, pred_labels, true_labels)

        # Compute final metrics
        precision = precision_metric.compute()
        recall = recall_metric.compute()
        f1 = f1_metric.compute()
        mAP = iou_metric.compute()
        
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Mean Average Precision (mAP): {mAP}")

        # Visualize metrics using bar charts
        metrics = {
            'Precision': precision.item(),
            'Recall': recall.item(),
            'F1 Score': f1.item(),
            'mAP': mAP['map'].item()  # mAP value
        }

        plot_metrics(metrics)


def plot_metrics(metrics):
    """Plot a bar chart for the metrics."""
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.bar(labels, values)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Object Detection Model Metrics')
    plt.show()
