import argparse
import io
import numpy as np
import torch
from PIL import Image

from _util.util_v1 import *  # I, a2bg, resize_min, etc.
import _util.util_v1 as uutil
from _util.pytorch_v1 import *

# import _util.pytorch_v1 as utorch  (alias unused, star import provides needed functions)
from _util.twodee_v0 import *  # cropbox, resize_square_dry, etc.
import _util.twodee_v0 as u2d
import _util.keypoints_v0 as ukey


# ------------------------ SEGMENTATION ------------------------
def infer_segmentation(segmenter, images, bbox_thresh=0.5, return_more=True):
    """
    Run background segmentation on a list of PIL images.
    Returns list of dicts with 'segmentation' (Image) and 'bbox' ([(x,y),(w,h)]).
    """
    anss = []
    _size = segmenter.hparams.largs.bg_seg.size
    segmenter.eval()
    for img in images:
        oimg = img
        img_proc = (
            I(img).resize_min(_size).convert("RGBA").alpha_bg(1).convert("RGB").pil()
        )
        timg = TF.to_tensor(img_proc)[None].to(segmenter.device)
        with torch.no_grad():
            out = segmenter(timg)
        mask = TF.to_pil_image(out["softmax"][0, 1].float().cpu()).resize(
            oimg.size[::-1]
        )
        ans = {"segmentation": I(mask)}
        # compute bbox
        a = np.array(ans["segmentation"].np()[-1] > bbox_thresh)
        x_any = np.any(a, axis=1).nonzero()[0]
        y_any = np.any(a, axis=0).nonzero()[0]
        if len(x_any) == 0 or len(y_any) == 0:
            bbox = [(0, 0), oimg.size]
        else:
            xmin, xmax = int(x_any.min()), int(x_any.max())
            ymin, ymax = int(y_any.min()), int(y_any.max())
            bbox = [(xmin, ymin), (xmax - xmin + 1, ymax - ymin + 1)]
        ans["bbox"] = bbox
        anss.append(ans)
    return anss


# ------------------------ POSE ESTIMATION ------------------------
def infer_pose(model_pose, segmenter, images, smoothing=0.1, pad_factor=1):
    """
    Run pose estimation on a list of PIL images, given a loaded pose model and segmenter.
    Returns list of dicts with keys: 'segmentation_output','bbox','cropbox','cropbox_inverse','input_image','out','keypoints'.
    """
    model_pose.eval()
    try:
        largs = model_pose.hparams.largs.adds_keypoints
    except AttributeError:
        largs = model_pose.hparams.largs.danbooru_coco
    _s = largs.size
    _p = _s * largs.padding
    anss = []
    segs = infer_segmentation(segmenter, images)
    for img, seg in zip(images, segs):
        oimg = img
        ans = {"segmentation_output": seg}
        bbox = seg["bbox"]
        cb = u2d.cropbox_sequence(
            [
                [bbox[0], bbox[1], bbox[1]],
                resize_square_dry(bbox[1], _s),
                [-_p * pad_factor / 2, _s + _p * pad_factor, _s],
            ]
        )
        icb = u2d.cropbox_inverse(oimg.size, *cb)
        cropped = u2d.cropbox(oimg, *cb).convert("RGBA").alpha(0).convert("RGB")
        ans.update(
            {
                "bbox": bbox,
                "cropbox": cb,
                "cropbox_inverse": icb,
                "input_image": cropped,
            }
        )
        timg = cropped.tensor()[None].to(model_pose.device)
        with torch.no_grad():
            out = model_pose(timg, smoothing=smoothing, return_more=True)
        ans["out"] = out
        kps = out["keypoints"][0].cpu().numpy()
        kps = u2d.cropbox_points(kps, *icb)
        ans["keypoints"] = kps
        anss.append(ans)
    return anss


# ------------------------ MODEL LOADER & RUNNER ------------------------
def load_model(ckpt_path: str):
    """
    Load and return (pose_model, segmenter) tuple from checkpoint paths.
    """
    # Add pytorch_lightning alias 'pl' for segmenter module
    # This is a workaround for loading a pytorch-lightning checkpoint that
    # was saved in an environment where `pl` was a global name. It should be kept.
    import builtins
    import pytorch_lightning as pl

    builtins.pl = pl

    # background segmenter
    from _train.character_bg_seg.models.alaska import Model as CharacterBGSegmenter

    segmenter = CharacterBGSegmenter.load_from_checkpoint(
        "./_train/character_bg_seg/runs/eyeless_alaska_vulcan0000/checkpoints/"
        "epoch=0096-val_f1=0.9508-val_loss=0.0483.ckpt"
    )
    # pose estimator
    if "feat_concat" in ckpt_path:
        from _train.character_pose_estim.models.passup import (
            Model as CharacterPoseEstimator,
        )
    elif "feat_match" in ckpt_path:
        from _train.character_pose_estim.models.fermat import (
            Model as CharacterPoseEstimator,
        )
    else:
        raise ValueError("Checkpoint name must include feat_concat or feat_match")
    pose_model = CharacterPoseEstimator.load_from_checkpoint(ckpt_path, strict=False)
    return pose_model, segmenter


def run_pose_estimation(model_tuple, buf: bytes, smoothing=0.1, pad_factor=1):
    """
    Inference API: takes (pose_model, segmenter) tuple and raw image bytes.
    Returns numpy array of keypoints.
    """
    pose_model, segmenter = model_tuple
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    results = infer_pose(pose_model, segmenter, [img], smoothing, pad_factor)
    return results[0]["keypoints"]


# ------------------------ CLI ENTRYPOINT ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fn_img", help="Path to input image")
    parser.add_argument("fn_model", help="Path to pose model checkpoint")
    args = parser.parse_args()
    # load models
    model_tuple = load_model(args.fn_model)
    # read image bytes
    with open(args.fn_img, "rb") as f:
        buf = f.read()
    # run inference
    ans_list = infer_pose(
        model_tuple[0], model_tuple[1], [Image.open(io.BytesIO(buf)).convert("RGB")]
    )
    bbox = ans_list[0]["bbox"]
    print(f"bounding box: {bbox}")
    print("keypoints:")
    for k, (x, y) in zip(ukey.coco_keypoints, ans_list[0]["keypoints"]):
        print(f"  {k}: ({x:.2f}, {y:.2f})")
    # save visualization
    from _scripts.pose_estimator import (
        _visualize,
    )  # reuse existing visualization if defined

    viz = _visualize(
        Image.open(io.BytesIO(buf)).convert("RGB"), bbox, ans_list[0]["keypoints"]
    )
    viz.save("./_samples/character_pose_estim.png")
    print("Output image saved to ./_samples/character_pose_estim.png")
