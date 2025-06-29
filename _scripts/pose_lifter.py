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
        dense=False,
    ):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, "Only odd filter widths are supported"

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_conv = nn.Conv1d(
            num_joints_in * in_features, channels, filter_widths[0], bias=False
        )
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append(
                (filter_widths[i] // 2 * next_dilation) if causal else 0
            )

            layers_conv.append(
                nn.Conv1d(
                    channels,
                    channels,
                    filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                    dilation=next_dilation if not dense else 1,
                    bias=False,
                )
            )
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(
                self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x)))
            )

        x = self.shrink(x)

        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)

        return x

    def receptive_field(self):
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames


class Lifter:
    def __init__(self, ckpt_path="/root/pretrained_h36m_cpn.bin"):
        # The `-arc 3,3,3,3,3` flag in the README corresponds to these filter widths
        filter_widths = [3, 3, 3, 3, 3]

        # Instantiate the verbatim model architecture
        self.model = TemporalModel(
            num_joints_in=17,
            in_features=2,
            num_joints_out=17,
            filter_widths=filter_widths,
            causal=False,
            dropout=0.25,
            channels=1024,
        )

        self.receptive_field = self.model.receptive_field()

        state_dict = torch.load(ckpt_path, map_location="cpu")
        # The model weights are nested in the checkpoint file under 'model_pos'
        self.model.load_state_dict(state_dict["model_pos"], strict=True)
        self.model.eval()

        self.device = "cpu"

    def to_gpu(self):
        self.model.to("cuda")
        self.device = "cuda"

    @torch.no_grad()
    def lift(
        self, coco_keypoints_2d: np.ndarray, image_width: int, image_height: int
    ) -> np.ndarray:
        h36m_kpts = coco_to_h36m(coco_keypoints_2d)
        root = h36m_kpts[0].copy()
        h36m_kpts_normalized = (h36m_kpts - root) / float(image_height)

        pad = (self.receptive_field - 1) // 2

        # Start with a single frame of shape (1, 17, 2)
        single_frame_sequence = h36m_kpts_normalized[np.newaxis, :]

        # Pad the frames dimension (axis 0) to the receptive field size.
        # The pad_width tuple must have a length equal to the input array's ndim (3).
        padded_sequence = np.pad(
            single_frame_sequence,
            ((pad, pad), (0, 0), (0, 0)),  # Padding for (frames, joints, features)
            "edge",
        )
        # padded_sequence now has shape (243, 17, 2)

        # Add the batch dimension at the beginning for the model.
        input_kpts = padded_sequence[np.newaxis, :]
        # input_kpts now has shape (1, 243, 17, 2)

        input_tensor = torch.from_numpy(input_kpts.astype("float32")).to(self.device)
        predicted_3d_pos_normalized = self.model(input_tensor).squeeze(0).cpu().numpy()

        # The model outputs a sequence; for single-image inference, we take the center frame
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
            final_result[coco_idx, 2] = (
                output_3d_h36m_normalized[h36m_idx, 2] * image_height
            )

        # Use nose depth for eyes and ears, as they are not in the H36M skeleton
        nose_z = final_result[0, 2]
        for idx in [1, 2, 3, 4]:
            final_result[idx, 2] = nose_z

        return final_result
