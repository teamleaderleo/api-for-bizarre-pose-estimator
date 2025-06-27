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
    # Install the specific PyTorch, Torchvision, and Detectron2 versions from the Dockerfile
    # This is for compatibility with the pretrained models.
    .run_commands(
        "pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html",
        # point directly to the correct wheel for Python 3.9
        "pip install 'detectron2 @ https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/detectron2-0.6-cp39-cp39-linux_x86_64.whl'",
    )
    # Install all other Python dependencies with pinned versions
    .pip_install(
        "matplotlib==3.5.0",
        "scipy==1.7.1",
        "scikit-learn==1.0.1",
        "scikit-image==0.18.3",
        "imagesize==1.3.0",
        "patool==1.12",
        "easydict==1.9",
        "igl==2.2.1",
        "meshplot==0.4.0",
        "Pillow==8.4.0",
        "wandb==0.12.7",
        "pyunpack==0.2.2",
        "opencv-contrib-python==4.5.4.60",
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
    scaledown_window=60
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
