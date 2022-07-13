

Use [mmpose](https://github.com/open-mmlab/mmpose) instead of [Lite-HRNet](https://github.com/HRNet/Lite-HRNet).

1. Modify mmpose/tools/deployment/pytorch2onnx.py to support dynamic batch:

```
    torch.onnx.export(
        model,
        one_img,
        output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=show,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["heatmap"],
        dynamic_axes={
            "image": {0: "batch"},
            "heatmap": {0: "batch"}
        })
```

2. Download pth model from [here](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#litehrnet-cvpr-2021). 
3. Convert pth model to ONNX model:

```
python tools/deployment/pytorch2onnx.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_30_coco_384x288.py litehrnet30_coco_384x288-a3aef5c4_20210626.pth --output-file litehrnet_30_coco_384x288-dynamic.onnx --verify --shape 1 3 384 288
```

4. Simplify ONNX model:

```
python -m onnxsim litehrnet_30_coco_384x288-dynamic.onnx litehrnet_30_coco_384x288-dynamic-sim.onnx 2 --dynamic-input-shape --input-shape image:1,3,384,288
```

5. Serialize TensorRT engine:

```
trtexec.exe --onnx=litehrnet_30_coco_384x288-dynamic-sim.onnx --saveEngine=litehrnet_30_coco_384x288-dynamic.trt --explicitBatch --minShapes=image:1x3x384x288  --optShapes=image:16x3x384x288  --maxShapes=image:64x3x384x288 --shapes=image:16x3x384x288 --workspace=1024
```

6. Run engine file with this project and get results below, it takes around 950ms with batch=64 on GTX 1050 Ti GPU, not as much fast as I think.

   ![](image/out0.jpg)