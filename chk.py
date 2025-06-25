import json

# Read the model definition
with open('extracted_model/usa_building_footprints.emd', 'r') as f:
    model_config = json.load(f)
    
print("Model Configuration:")
print(f"Framework: {model_config.get('Framework')}")
print(f"Model Type: {model_config.get('ModelType')}")
print(f"Input Size: {model_config.get('ImageHeight')} x {model_config.get('ImageWidth')}")
print(f"Bands: {model_config.get('ExtractBands')}")
print(f"Classes: {model_config.get('Classes')}")