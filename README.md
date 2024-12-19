# smooth_predictions_by_blending_patches_Pytorch
Edit version of smooth_predictions_by_blending_patches code from python_for_microscopists to run on Pytorch segmentation models 

# Original source code 

https://github.com/bnsreenu/python_for_microscopists/blob/master/229_smooth_predictions_by_blending_patches/229_prediction_aerial_imagery_using_smooth_blending.py


# Updated Prediction code to work with batch prediction 


```python
def batch_predict(model, data, batch_size=32):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_samples = data.shape[0]
    predictions = []
  
    with torch.no_grad():  # Disable gradient calculation for inference
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = data[start_idx:end_idx]

            # Convert to PyTorch tensor and move to the appropriate device
            batch_tensor = torch.from_numpy(batch_data).float().to(device)

            # Perform the forward pass
            batch_predictions = model(batch_tensor)
            for pred in batch_predictions:
              # Move predictions to CPU and convert to NumPy array
              pred = pred.cpu().numpy().transpose(1,2,0)
              predictions.append(pred)

    return np.array(predictions)
```
