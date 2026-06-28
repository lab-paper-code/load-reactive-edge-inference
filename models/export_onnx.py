"""Regenerate the three public image classifiers as ONNX (224x224, batch 1).
Requires: torch, timm, onnx. Run: python3 export_onnx.py
"""
import torch, timm
SPECS = {
    "mobilenet-v2-050": "mobilenetv2_050",
    "mobilenet-v2-100": "mobilenetv2_100",
    "efficientnet-b4":  "efficientnet_b4",
}
def main():
    for out, tm in SPECS.items():
        m = timm.create_model(tm, pretrained=True).eval()
        x = torch.randn(1, 3, 224, 224)
        torch.onnx.export(m, x, f"{out}.onnx", input_names=["input"],
                          output_names=["logits"], opset_version=13,
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
        print("wrote", out + ".onnx")
if __name__ == "__main__":
    main()
