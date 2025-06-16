import modal
from fastapi import UploadFile, File, HTTPException

# 1) Define Modal app
app = modal.App("bizarre-pose-api")

# 2) Build image with only deps we actually need to import at runtime
image = (
    modal.Image.debian_slim(python_version="3.9")
    # System libs for headless OpenCV
    .run_commands(
        [
            "apt-get update",
            "DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "libgl1-mesa-glx libsm6 libxrender1 wget",  # add wget
            # pre-cache the DeepLabV3 COCO backbone so inference never downloads it
            "mkdir -p /root/.cache/torch/hub/checkpoints",
            "wget "
            "https://download.pytorch.org/models/"
            "deeplabv3_resnet101_coco-586e9e4e.pth "
            "-O /root/.cache/torch/hub/checkpoints/"
            "deeplabv3_resnet101_coco-586e9e4e.pth",
        ]
    )
    # Python deps: torch, kornia, scikit-image, lightning, FastAPI, + multipart support
    .pip_install(
        "torch",
        "torchvision",
        "opencv-contrib-python==4.5.4.60",
        "Pillow==8.4.0",
        "scikit-image==0.18.3",
        "kornia==0.6.2",
        "pytorch-lightning>=1.5.0",
        "fastapi",
        "python-multipart",
    )
    # Include inference scripts as a Python package
    .add_local_python_source("_scripts")
    # Include its utilities as a Python package
    .add_local_python_source("_util")
    # Background segmenter code + checkpoint
    .add_local_dir(
        "_train/character_bg_seg/models",
        "/root/_train/character_bg_seg/models",
    )
    .add_local_file(
        "_train/character_bg_seg/runs/eyeless_alaska_vulcan0000/checkpoints/epoch=0096-val_f1=0.9508-val_loss=0.0483.ckpt",
        "/root/_train/character_bg_seg/runs/eyeless_alaska_vulcan0000/checkpoints/epoch=0096-val_f1=0.9508-val_loss=0.0483.ckpt",
    )
    # Pose-estimator code + checkpoint
    .add_local_dir(
        "_train/character_pose_estim/models",
        "/root/_train/character_pose_estim/models",
    )
    .add_local_file(
        "_train/character_pose_estim/runs/feat_match+data.ckpt",
        "/root/_train/character_pose_estim/runs/feat_match+data.ckpt",
    )
    # Danbooru tagger code + checkpoint
    .add_local_dir(
        "_train/danbooru_tagger",
        "/root/_train/danbooru_tagger",
    )
    # Add the rule‐book JSONs:
    .add_local_dir("_data/danbooru/_filters", "/root/_data/danbooru/_filters")
)


# 3) Define GPU-backed class; no mounts needed
@app.cls(
    # Try T4 first, then...
    gpu=["T4", "L4", "A10G", "L40S", "A100", "any"],
    image=image,
    scaledown_window=60
)
class BizarrePoseModel:
    @modal.enter()
    def load_model_once(self):
        import sys, types

        # Stub out Detectron2
        detectron2_pkg = types.ModuleType("detectron2")
        detectron2_cfg = types.ModuleType("detectron2.config")
        # supply a dummy get_cfg() so fermat.py won't crash
        detectron2_cfg.get_cfg = lambda: types.SimpleNamespace(
            MODEL=types.SimpleNamespace(
                KEYPOINT_ON=False,
                WEIGHTS="",
                ROI_HEADS=types.SimpleNamespace(NUM_CLASSES=0),
            )
        )
        detectron2_pkg.config = detectron2_cfg
        sys.modules["detectron2"] = detectron2_pkg
        sys.modules["detectron2.config"] = detectron2_cfg

        sys.path.insert(0, "/root")
        from _scripts.pose_estimator import load_model, run_pose_estimation

        ckpt = "/root/_train/character_pose_estim/runs/feat_match+data.ckpt"
        self.model = load_model(ckpt)
        self.run = run_pose_estimation

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def predict(self, file: UploadFile = File(...)):
        if not file.content_type.startswith("image/"):
            raise HTTPException(415, "Please upload an image file")
        buf = await file.read()
        try:
            keypoints = self.run(self.model, buf)
        except Exception as e:
            raise HTTPException(500, f"Inference failed: {e}")
        return {"keypoints": keypoints}


if __name__ == "__main__":
    app.deploy()
