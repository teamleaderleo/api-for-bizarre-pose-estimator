import modal
from fastapi import UploadFile, File, HTTPException

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
        # `igl` and `meshplot` are removed. Our deep dive through every single
        # model file confirmed they are unused, optional dependencies. The API's
        # code path for segmentation and pose estimation never calls them.
        # Let versions chosen by detectron2 take precedence
        "Pillow",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "matplotlib",
        # Keep pins for the less common or more specific packages from your Dockerfile
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
    .add_local_dir("_data", "/root/_data")
    .add_local_python_source("_scripts")
    .add_local_python_source("_util")
)


# 3) Define GPU-backed class with simplified loading logic
@app.cls(
    # Try T4 first, then...
    gpu=["T4", "L4", "A10G", "L40S", "A100", "any"],
    image=image,
    scaledown_window=60,
)
class BizarrePoseModel:
    @modal.enter()
    def load_model_once(self):
        """
        This runs once per container startup. We load the models using the
        project's own loader script, without any stubs or monkey-patching.
        """
        import sys

        # Add the project root to the Python path to allow imports like `_scripts...`
        sys.path.insert(0, "/root")

        from _scripts.pose_estimator import load_model

        # The path to the checkpoint inside the Modal container
        ckpt_path = "/root/_train/character_pose_estim/runs/feat_match+data.ckpt"

        print("Loading models from checkpoint...")
        # load_model returns a (pose_model, segmenter) tuple
        self.model_tuple = load_model(ckpt_path)
        print("Models loaded successfully.")

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
            # Call the original inference function with the loaded models
            keypoints_array = run_pose_estimation(self.model_tuple, buf)
        except Exception as e:
            # Log the full exception for better debugging
            import traceback

            traceback.print_exc()
            raise HTTPException(500, f"Inference failed: {e}")

        # Create the structured dictionary by zipping names with coordinates
        keypoints_dict = {
            name: coords.tolist()  # .tolist() converts numpy array to python list
            for name, coords in zip(ukey.coco_keypoints, keypoints_array)
        }

        return {"keypoints": keypoints_dict}


if __name__ == "__main__":
    app.deploy()
