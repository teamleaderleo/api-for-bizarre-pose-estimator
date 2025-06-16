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


# 3) Define GPU-backed class
@app.cls(
    # Try T4 first, then...
    gpu=["T4", "L4", "A10G", "L40S", "A100", "any"],
    image=image,
    scaledown_window=60
)
class BizarrePoseModel:
    @modal.enter()
    def load_model_once(self):
        import sys, torch, importlib

        # 1) Stub out the Danbooru tagger JSON loader
        import _util.util_v1 as uutil
        uutil.jread = lambda fn, mode="r": {}

        # 2) Define a dummy PretrainedKeypointDetector
        class DummyPKD(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def eval(self): return self
            def parameters(self): return []
            def forward(self, img, return_more=False):
                bs, _, h, w = img.shape
                zero_hm = torch.zeros(bs, 17+8, h, w, device=img.device)
                return {'keypoint_heatmaps': zero_hm}

        # 3) Monkey‐patch ResnetFeatureExtractor to always use torchvision
        import torchvision.transforms as T
        import torchvision.models as tv_models
        class TorchvisionResnetExtractor(torch.nn.Module):
            def __init__(self, inferserve_query):
                super().__init__()
                # load standard ResNet50
                resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
                self.resize = T.Resize(256)
                self.prep   = T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225],
                )
                self.conv1   = resnet.conv1
                self.bn1     = resnet.bn1
                self.relu    = resnet.relu
                self.maxpool = resnet.maxpool
                self.layer1  = resnet.layer1
                self.layer2  = resnet.layer2
                self.layer3  = resnet.layer3

            def forward(self, x):
                ans = {}
                x = self.resize(x)
                x = self.prep(x)
                x = self.conv1(x); ans['conv1'] = x
                x = self.bn1(x);   x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x); ans['layer1'] = x
                x = self.layer2(x); ans['layer2'] = x
                x = self.layer3(x); ans['layer3'] = x
                return ans

        # Unload and reimport the model modules so we can patch them
        for mod in ("_train.character_pose_estim.models.fermat",
                    "_train.character_pose_estim.models.passup"):
            sys.modules.pop(mod, None)
        fermat = importlib.import_module("_train.character_pose_estim.models.fermat")
        passup = importlib.import_module("_train.character_pose_estim.models.passup")

        fermat.PretrainedKeypointDetector      = DummyPKD
        passup.PretrainedKeypointDetector      = DummyPKD
        fermat.ResnetFeatureExtractor          = TorchvisionResnetExtractor
        passup.ResnetFeatureExtractor          = TorchvisionResnetExtractor

        # 4) Load inference library
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
