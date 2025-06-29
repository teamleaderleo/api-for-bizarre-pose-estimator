import modal
from fastapi import UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np

# 1) Define Modal app
app = modal.App("bizarre-pose-api")

# 2) Build a complete image based on the original project's Dockerfile.
# This ensures all dependencies, including Detectron2, are correctly installed.
image = (
    modal.Image.debian_slim(python_version="3.9")
    # System libraries required for OpenCV, Pillow, and other ML packages
    .run_commands(
        [
            "apt-get update",
            "DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "wget cmake ffmpeg libgl1-mesa-glx libsm6 libxext6 libxrender-dev",
        ]
    )
    # Pre-cache ALL models during the build ***
    .run_commands(
        "mkdir -p /root/.cache/torch/hub/checkpoints",
        # 1. DeepLabV3 for the segmenter
        "wget https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth -O /root/.cache/torch/hub/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth",
        # 2. ResNet50 for the pose estimator's backbone
        "wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth",
        # 3. Detectron2's pretrained model
        "wget https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl -O /root/model_final_997cc7.pkl",
        # 4. The 2D to 3D pose lifter model
        "wget https://dl.fbaipublicfiles.com/pose-baseline/pretrained_h36m_2d_to_3d.bin -O /root/pretrained_h36m_2d_to_3d.bin",
    )
    # Install Torch, Torchvision, and Detectron2 first.
    # We must use run_commands here because of the --find-links (-f) flag.
    .run_commands(
        # First, install the correct PyTorch and Torchvision versions
        "pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html",
        # Now, install Detectron2 by pointing pip to the official index URL.
        # This is more robust than linking to a specific file.
        "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html",
    )
    # Install all other Python dependencies.
    .pip_install(
        "Pillow==8.4.0",
        "protobuf==3.20.3",
        # The error log shows a conflict between old torch (built against NumPy 1.x)
        # and the new NumPy 2.x. We pin NumPy to a compatible 1.x version.
        "numpy==1.21.5",
        # `igl` and `meshplot` are removed. Our deep dive through every single
        # model file confirmed they are unused, optional dependencies. The API's
        # code path for segmentation and pose estimation never calls them.
        # The runtime error shows that the default `torchmetrics` installed by pip
        # is too new for our old `torchvision`. We pin it to an older, compatible
        # version to resolve the `VGG16_Weights` ImportError.
        "torchmetrics==0.6.2",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "matplotlib",
        # Keep pins for the less common or more specific packages from our Dockerfile
        "easydict==1.9",
        "imagesize==1.3.0",
        "patool==1.12",
        "wandb==0.12.7",
        "pyunpack==0.2.2",
        "kornia==0.6.2",
        "pytorch-lightning==1.3.8",
        # Dependencies for the web endpoint itself
        "fastapi",
        "python-multipart",
    )
    # Add the project's code and data directories wholesale.
    # No more cherry-picking files, which avoids missing dependencies.
    .add_local_dir("_train", "/root/_train")
    .add_local_python_source("_scripts")
    .add_local_python_source("_util")
    # Instead of uploading the entire `_data` directory (with gigabytes of raw images),
    # we only include the small subdirectory of JSON rule files that the models
    # actually need to load at runtime. This will fix the upload timeout.
    .add_local_dir("_data/danbooru/_filters", "/root/_data/danbooru/_filters")
)


# 3) Define GPU-backed class with simplified loading logic
@app.cls(
    # Try T4 first, then...
    gpu=["T4", "L4", "A10G", "L40S", "A100", "any"],
    image=image,
    scaledown_window=20,
    timeout=60,
    enable_memory_snapshot=True,  # *** PERFORMANCE FIX: Enable memory snapshots ***
)
class BizarrePoseModel:
    # *** STAGE 1: Load models onto CPU. This is snapshotted. ***
    @modal.enter(snap=True)
    def load_models_to_cpu(self):
        import sys

        # Add the project root to the Python path to allow imports like `_scripts...`
        sys.path.insert(0, "/root")
        from _scripts.pose_estimator import load_model
        from _scripts.pose_lifter import SimpleLifter

        print("Loading 2D pose models to CPU...")
        self.model_tuple = load_model(
            "/root/_train/character_pose_estim/runs/feat_match+data.ckpt"
        )
        print("Loading 3D lifter model to CPU...")
        self.lifter = SimpleLifter()
        print("CPU models loaded successfully.")

    # *** STAGE 2: Move models to GPU. This runs after restoring from snapshot. ***
    @modal.enter(snap=False)
    def move_models_to_gpu(self):
        print("Moving 2D pose models to GPU...")
        pose_model, segmenter = self.model_tuple
        pose_model.to("cuda")
        segmenter.to("cuda")
        print("Models moved to GPU successfully.")

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def predict(self, file: UploadFile = File(...)):
        """
        The inference endpoint. It uses the project's `run_pose_estimation`
        function with the fully loaded models.
        """
        import sys

        # Ensure project root is on path for this worker too
        sys.path.insert(0, "/root")
        import _util.keypoints_v0 as ukey
        from _scripts.pose_estimator import run_pose_estimation

        if not file.content_type.startswith("image/"):
            raise HTTPException(415, "Please upload an image file.")
        buf = await file.read()

        try:
            img_h = Image.open(io.BytesIO(buf)).height
            keypoints_2d_array = run_pose_estimation(self.model_tuple, buf)

            # The lifter now handles all mapping and returns a (17, 3) array
            # with [original_x, original_y, inferred_z] in the correct COCO order.
            final_keypoints_3d = self.lifter.lift(
                keypoints_2d_array.astype("float32"), img_h
            )

        except Exception as e:
            # Log the full exception for better debugging
            import traceback

            traceback.print_exc()
            raise HTTPException(500, f"Inference failed: {e}")

        # Create the structured dictionary by zipping names with coordinates
        keypoints_dict = {
            name: coords.tolist()
            for name, coords in zip(ukey.coco_keypoints, final_keypoints_3d)
        }
        return {"keypoints": keypoints_dict}


if __name__ == "__main__":
    app.deploy()
