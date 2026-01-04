import sys
sys.path.insert(0, '.')  # Adjust path as needed
from src.models.builders import create_multimodal_model

input_shapes = {'metadata': (3,)}
selected_modalities = ['metadata']
class_weights = {0: 1.0, 1: 1.0, 2: 1.0}
model = create_multimodal_model(input_shapes, selected_modalities, class_weights, strategy=None)

print("\n" + "="*80)
print("ARCHITECTURE INSPECTION - METADATA-ONLY")
print("="*80)
model.summary()
print("\nCRITICAL CHECK:")
dense_layers = [l for l in model.layers if 'Dense' in l.__class__.__name__]
print(f"Dense layers: {len(dense_layers)}")
output_layer = model.layers[-1]
print(f"Output layer: {output_layer.__class__.__name__}")
if 'Activation' in output_layer.__class__.__name__:
    print("✅ PASS: RF quality preserved (Activation layer, no Dense)")
else:
    print("❌ FAIL: Dense layer may degrade RF")
