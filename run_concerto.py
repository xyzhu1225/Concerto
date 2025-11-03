# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import numpy as np
import concerto
import torch
import torch.nn as nn
import open3d as o3d
import argparse
import trimesh
try:
    import flash_attn
except ImportError:
    flash_attn = None

# ScanNet Meta data
VALID_CLASS_IDS_20 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)


CLASS_LABELS_20 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

CLASS_COLOR_20 = [SCANNET_COLOR_MAP_20[id] for id in VALID_CLASS_IDS_20]


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)

# Updated script with additional argparse parameters
# (Starting from the original user-provided code; edits needed below)

class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="semantic segmentation inference")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input PLY point cloud")
    parser.add_argument("--output_path", type=str, required=True, help="Output base path (no extension)")
    parser.add_argument("--model_path", type=str, default="Pointcept/Concerto", help="Concerto model repo or path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument('--wo_color', dest='wo_color', action='store_true', help="disable the color.")
    parser.add_argument('--wo_normal', dest='wo_normal', action='store_true', help="disable the normal.")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # set random seed
    concerto.utils.set_seed(46647087)

    # Load model
    if flash_attn is not None:
        model = concerto.load("concerto_large", repo_id=args.model_path).to(device)
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,
        )
        model = concerto.load(
            "concerto_large", repo_id=args.model_path, custom_config=custom_config
        ).to(device)

    # Load linear probing seg head
    ckpt = concerto.load(
        "concerto_large_linear_prob_head_sc",
        repo_id=args.model_path,
        ckpt_only=True,
    )
    seg_head = SegHead(**ckpt["config"]).to(device)
    seg_head.load_state_dict(ckpt["state_dict"])

    # Load default data transform pipeline
    transform = concerto.transform.default()

    # Load input PLY
    print(f"Loading point cloud from {args.input_path} ...")
    pc = trimesh.load(args.input_path)
    coords = np.asarray(pc.vertices)

    if hasattr(pc, "colors") and pc.colors is not None and len(pc.colors) == len(pc.vertices):
        colors = np.asarray(pc.colors[:, :3], dtype=np.uint8)
        print("Using existing colors from the point cloud.")
    else:
        colors = np.zeros_like(coords, dtype=np.uint8)

    point = {
        "coord": coords.astype(np.float32),
        "color": colors.astype(np.float32),
    }

    if args.wo_color:
        point["color"] = np.zeros_like(point["coord"])
    if args.wo_normal:
        point["normal"] = np.zeros_like(point["coord"])

    original_coord = point["coord"].copy()
    point = transform(point)

    # Inference
    model.eval()
    seg_head.eval()
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and "cuda" in device:
                point[key] = point[key].to(device, non_blocking=True)
        point = model(point)
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        feat = point.feat
        seg_logits = seg_head(feat)
        pred = seg_logits.argmax(dim=-1).data.cpu().numpy()
        color = np.array(CLASS_COLOR_20)[pred]

    # Visualize & Save
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point.coord.cpu().detach().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color / 255.0)
    o3d.io.write_point_cloud(f"{args.output_path}.ply", pcd)
    print(f"Saved: {args.output_path}.ply")
