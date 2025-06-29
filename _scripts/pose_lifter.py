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


# --- Model Architecture from VideoPose3D ---
class TemporalModel(nn.Module):
    def __init__(
        self,
        num_joints_in,
        in_features,
        num_joints_out,
        filter_widths,
        causal=False,
        dropout=0.25,
        channels=1024,
    ):
        super().__init__()

        self.num_joints_in = num_joints_in
        self.in_features = in_features

        # "This is the corrected architecture based on the original source code"
        self.expand_conv = nn.Conv1d(
            num_joints_in * in_features, channels, filter_widths[0], bias=False
        )
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)

        self.layers_conv = nn.ModuleList()
        self.layers_bn = nn.ModuleList()

        # Build the main temporal blocks
        for i in range(1, len(filter_widths)):
            dilation = (filter_widths[0] - 1) ** (i - 1)
            self.layers_conv.append(
                nn.Conv1d(
                    channels, channels, filter_widths[i], dilation=dilation, bias=False
                )
            )
            self.layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

        # Final layer, named 'shrink' to match the state_dict
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        # Input shape: (batch, frames, joints, features)
        x = x.permute(0, 3, 1, 2).contiguous()  # -> (batch, features, frames, joints)
        x = x.view(
            x.shape[0], x.shape[1], x.shape[2], self.num_joints_in
        )  # Sanity check shape
        x = x.view(
            x.shape[0], self.in_features * self.num_joints_in, x.shape[2]
        )  # -> (batch, joints*features, frames)

        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.layers_conv)):
            res = x
            x = self.drop(self.relu(self.layers_bn[i](self.layers_conv[i](x))))
            x = res + x  # Residual connection

        x = self.shrink(x)

        # Output shape: (batch, frames, joints, 3)
        x = (
            x.view(x.shape[0], 3, -1, self.num_joints_in)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        return x


class Lifter:
    def __init__(self, ckpt_path="/root/pretrained_h36m_cpn.bin"):
        # This architecture matches the `-arc 3,3,3,3,3` model with 243-frame receptive field
        filter_widths = [3, 3, 3, 3, 3]

        self.model = TemporalModel(
            num_joints_in=17,
            in_features=2,
            num_joints_out=17,
            filter_widths=filter_widths,
            causal=False,
            dropout=0.25,
            channels=1024,
        )

        # Load the pretrained model weights
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # Rename keys from the checkpoint to match our model structure
        # (original code used 'model_pos.layers' which is flattened by state_dict)
        for k in list(state_dict["model_pos"].keys()):
            new_key = k.replace("model_pos.", "")
            state_dict["model_pos"][new_key] = state_dict["model_pos"].pop(k)

        self.model.load_state_dict(state_dict["model_pos"], strict=True)
        self.model.eval()

        # Calculate the receptive field
        self.receptive_field = 1
        for i in range(1, len(filter_widths)):
            self.receptive_field += (filter_widths[i] - 1) * (
                (filter_widths[0] - 1) ** (i - 1)
            )

        self.device = "cpu"

    def to_gpu(self):
        self.model.to("cuda")
        self.device = "cuda"

    @torch.no_grad()
    def lift(
        self, coco_keypoints_2d: np.ndarray, image_width: int, image_height: int
    ) -> np.ndarray:
        h36m_kpts = coco_to_h36m(coco_keypoints_2d)

        # Normalize for the model
        root = h36m_kpts[0].copy()
        h36m_kpts_normalized = (h36m_kpts - root) / float(image_height)

        # Pad the single frame to the model's full receptive field
        pad = (self.receptive_field - 1) // 2
        input_kpts = np.pad(
            h36m_kpts_normalized[np.newaxis, np.newaxis],
            ((0, 0), (0, 0), (pad, pad), (0, 0)),
            "edge",
        )
        input_kpts = np.transpose(input_kpts, (0, 2, 3, 1))

        # Run inference
        input_tensor = torch.from_numpy(input_kpts.astype("float32")).to(self.device)
        predicted_3d_pos_normalized = self.model(input_tensor).squeeze(0).cpu().numpy()

        output_3d_h36m_normalized = predicted_3d_pos_normalized[pad]

        # Denormalize Z coordinate and map back to COCO format
        final_result = np.zeros((17, 3), dtype=np.float32)
        final_result[:, :2] = coco_keypoints_2d

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
            # Denormalize Z using image height
            final_result[coco_idx, 2] = (
                output_3d_h36m_normalized[h36m_idx, 2] * image_height
            )

        # Use nose depth for eyes and ears
        nose_z = final_result[0, 2]
        for idx in [1, 2, 3, 4]:
            final_result[idx, 2] = nose_z

        return final_result
