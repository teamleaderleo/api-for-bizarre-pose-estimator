import io
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
            "DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libsm6 libxrender1",
        ]
    )
    # Python deps: torch, kornia, scikit-image, lightning, and FastAPI
    .pip_install(
        "torch",
        "torchvision",
        "opencv-contrib-python==4.5.4.60",
        "Pillow==8.4.0",
        "scikit-image==0.18.3",
        "kornia==0.6.2",
        "pytorch-lightning==1.3.8",
        "fastapi",
    )
    # Include inference scripts as a Python package
    .add_local_python_source("_scripts")
    # Include extracted checkpoint directory
    .add_local_dir("_train", "/root/_train")
)


# 3) Define GPU-backed class; no mounts needed
@app.cls(
    # Try T4 first, then A10G, then any other available GPU
    gpu=["t4", "a10", "any"],
    image=image,
)
class BizarrePoseModel:
    def __enter__(self):
        import sys

        sys.path.insert(0, "/root")
        from _scripts.pose_estimator import load_model, run_pose_estimation

        ckpt = "/root/_train/character_pose_estim/runs/feat_concat+data.ckpt"
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
