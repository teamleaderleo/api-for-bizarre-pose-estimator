import torch
import torch.nn as nn
import numpy as np


def coco_to_h36m(coco_kpts: np.ndarray) -> np.ndarray:
    """
    Convert COCO 17 keypoints to H36M 17 keypoints format.
    Based on the official research-grade implementation from MotionBERT.
    """
    h36m_kpts = np.zeros((17, 2), dtype=np.float32)

    # COCO indices: 0-nose, 1-Leye, 2-Reye, ..., 16-Rankle
    # H36M indices: 0-root, 1-rhip, 2-rkne, ..., 16-rwri

    h36m_kpts[0] = (coco_kpts[11] + coco_kpts[12]) * 0.5  # root = avg(lhip, rhip)
    h36m_kpts[1] = coco_kpts[12]  # rhip
    h36m_kpts[2] = coco_kpts[14]  # rkne
    h36m_kpts[3] = coco_kpts[16]  # rank
    h36m_kpts[4] = coco_kpts[11]  # lhip
    h36m_kpts[5] = coco_kpts[13]  # lkne
    h36m_kpts[6] = coco_kpts[15]  # lank
    h36m_kpts[8] = (coco_kpts[5] + coco_kpts[6]) * 0.5  # neck = avg(lsho, rsho)
    h36m_kpts[7] = (h36m_kpts[0] + h36m_kpts[8]) * 0.5  # belly = avg(root, neck)
    h36m_kpts[9] = coco_kpts[0]  # nose
    h36m_kpts[10] = (coco_kpts[1] + coco_kpts[2]) * 0.5  # head = avg(leye, reye)
    h36m_kpts[11] = coco_kpts[5]  # lsho
    h36m_kpts[12] = coco_kpts[7]  # lelb
    h36m_kpts[13] = coco_kpts[9]  # lwri
    h36m_kpts[14] = coco_kpts[6]  # rsho
    h36m_kpts[15] = coco_kpts[8]  # relb
    h36m_kpts[16] = coco_kpts[10]  # rwri

    return h36m_kpts


class SimpleLifter(nn.Module):
    def __init__(self, ckpt_path="/root/pretrained_h36m_2d_to_3d.bin"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(34, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 51),
        )
        self.mlp.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.eval()

    @torch.no_grad()
    def lift(self, coco_keypoints_2d: np.ndarray, image_height: int) -> np.ndarray:
        # 1. Convert COCO keypoints to the H36M format the model expects.
        h36m_kpts = coco_to_h36m(coco_keypoints_2d)

        # 2. Normalize the H36M keypoints for the model.
        root = h36m_kpts[0].copy()
        h36m_kpts -= root
        h36m_kpts /= float(image_height)

        # 3. Predict the 3D pose in H36M space.
        input_tensor = torch.from_numpy(h36m_kpts.flatten()).float().unsqueeze(0)
        output_3d_h36m = self.mlp(input_tensor).view(17, 3).cpu().numpy()

        # 4. Create the final output array, preserving the original COCO X,Y coordinates.
        final_result = np.zeros((17, 3), dtype=np.float32)
        final_result[:, :2] = coco_keypoints_2d

        # 5. Map the inferred Z-coordinates from the H36M output back to the correct COCO joints.
        h36m_to_coco_map = {
            1: 12,
            2: 14,
            3: 16,  # Right Leg
            4: 11,
            5: 13,
            6: 15,  # Left Leg
            9: 0,  # Nose
            11: 5,
            12: 7,
            13: 9,  # Left Arm
            14: 6,
            15: 8,
            16: 10,  # Right Arm
        }
        for h36m_idx, coco_idx in h36m_to_coco_map.items():
            final_result[coco_idx, 2] = output_3d_h36m[h36m_idx, 2]

        # 6. Intelligently handle missing joints (eyes, ears) by using the nose's depth.
        # This is a key refinement for visual quality.
        nose_z = final_result[0, 2]
        final_result[1, 2] = nose_z  # Left Eye
        final_result[2, 2] = nose_z  # Right Eye
        final_result[3, 2] = nose_z  # Left Ear
        final_result[4, 2] = nose_z  # Right Ear

        return final_result
