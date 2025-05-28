import torchvision.models as models
import torch.nn as nn

def get_model(model_name='convnext_large', pretrained=True, num_classes=101):
    model_fn = getattr(models, model_name)
    
    weights_enum = models.get_model_weights(model_name)
    
    weights = weights_enum.DEFAULT if pretrained else None
    
    model = model_fn(weights=weights)
    
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features, 
        num_classes
    )
    
    return model