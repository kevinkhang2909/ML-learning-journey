from optimum.onnxruntime import ORTModelForImageClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoFeatureExtractor
from pathlib import Path


model_id = "nateraw/vit-base-beans"
onnx_path = Path("onnx")

# load vanilla transformers and convert to onnx
model = ORTModelForImageClassification.from_pretrained(model_id, from_transformers=True)
preprocessor = AutoFeatureExtractor.from_pretrained(model_id)

# save onnx checkpoint and tokenizer
model.save_pretrained(onnx_path)
preprocessor.save_pretrained(onnx_path)

# from transformers import pipeline
#
# vanilla_clf = pipeline("image-classification", model=model, feature_extractor=preprocessor)
# print(vanilla_clf("https://datasets-server.huggingface.co/assets/beans/--/default/validation/30/image/image.jpg"))

# create ORTQuantizer and define quantization configuration
dynamic_quantizer = ORTQuantizer.from_pretrained(model)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# apply the quantization configuration to the model
model_quantized_path = dynamic_quantizer.quantize(
    save_dir=onnx_path,
    quantization_config=dqconfig,
)

import os

# get model file size
size = os.path.getsize(onnx_path / "model.onnx")/(1024*1024)
quantized_model = os.path.getsize(onnx_path / "model_quantized.onnx")/(1024*1024)

print(f"Model file size: {size:.2f} MB")
print(f"Quantized Model file size: {quantized_model:.2f} MB")
#   Model file size: 330.27 MB
#   Quantized Model file size: 84.50 MB

from optimum.onnxruntime import ORTModelForImageClassification
from transformers import pipeline, AutoFeatureExtractor

model = ORTModelForImageClassification.from_pretrained(onnx_path, file_name="model_quantized.onnx")
preprocessor = AutoFeatureExtractor.from_pretrained(onnx_path)

q8_clf = pipeline("image-classification", model=model, feature_extractor=preprocessor)
print(q8_clf("https://datasets-server.huggingface.co/assets/beans/--/default/validation/30/image/image.jpg"))
