import os
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import logging
import copy
import datetime
import json
script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

from .vitpose_utils.utils import bbox_from_detector, crop, load_pose_metas_from_kp2ds_seq, aaposemeta_to_dwpose_scail
from .pose_draw.draw_pose_utils import draw_pose_to_canvas_np

def convert_openpose_to_target_format(frames, max_people=2):
    NUM_BODY = 18
    NUM_FACE = 70
    NUM_HAND = 21

    results = []
    for frame in frames:
        canvas_width = frame['canvas_width']
        canvas_height = frame['canvas_height']
        people = frame['people'][:max_people]

        bodies = []
        hands = []
        faces = []
        body_scores = []
        hand_scores = []
        face_scores = []

        for person in people:
            pose_raw = person.get('pose_keypoints_2d') or []
            if len(pose_raw) != NUM_BODY * 3:
                continue

            pose = np.array(pose_raw).reshape(-1, 3)
            pose_xy = np.stack([pose[:, 0] / canvas_width, pose[:, 1] / canvas_height], axis=1)
            bodies.append(pose_xy)
            body_scores.append(pose[:, 2])

            face_raw = person.get('face_keypoints_2d') or []
            if len(face_raw) == NUM_FACE * 3:
                face = np.array(face_raw).reshape(-1, 3)
                face_xy = np.stack([face[:, 0] / canvas_width, face[:, 1] / canvas_height], axis=1)
                faces.append(face_xy)
                face_scores.append(face[:, 2])

            hand_left_raw = person.get('hand_left_keypoints_2d') or []
            hand_right_raw = person.get('hand_right_keypoints_2d') or []
            if len(hand_left_raw) == NUM_HAND * 3:
                hand_left = np.array(hand_left_raw).reshape(-1, 3)
                hand_left_xy = np.stack([hand_left[:, 0] / canvas_width, hand_left[:, 1] / canvas_height], axis=1)
                hands.append(hand_left_xy)
                hand_scores.append(hand_left[:, 2])
            if len(hand_right_raw) == NUM_HAND * 3:
                hand_right = np.array(hand_right_raw).reshape(-1, 3)
                hand_right_xy = np.stack([hand_right[:, 0] / canvas_width, hand_right[:, 1] / canvas_height], axis=1)
                hands.append(hand_right_xy)
                hand_scores.append(hand_right[:, 2])

        result = {
            'bodies': {
                'candidate': np.array(bodies, dtype=np.float32),
                'subset': np.array([np.arange(NUM_BODY) for _ in bodies], dtype=np.float32) if bodies else np.array([])
            },
            'hands': np.array(hands, dtype=np.float32),
            'faces': np.array(faces, dtype=np.float32),
            'body_score': np.array(body_scores, dtype=np.float32),
            'hand_score': np.array(hand_scores, dtype=np.float32),
            'face_score': np.array(face_scores, dtype=np.float32)
        }
        results.append(result)
    return results

def scale_faces(poses, pose_2d_ref):
    # Input: two lists of dict, poses[0]['faces'].shape: 1, 68, 2  , poses_ref[0]['faces'].shape: 1, 68, 2
    # Scale the facial keypoints in poses according to the center point of the face
    # That is: calculate the distance from the center point (idx: 30) to other facial keypoints in ref,
    # and the same for poses, then get scale_n as the ratio
    # Clamp scale_n to the range 0.8-1.5, then apply it to poses
    # Note: poses are modified in place

    ref = pose_2d_ref[0]
    pose_0 = poses[0]

    face_0 = pose_0['faces']  # shape: (1, 68, 2)
    face_ref = ref['faces']

    # Extract numpy arrays
    face_0 = np.array(face_0[0])      # (68, 2)
    face_ref = np.array(face_ref[0])

    # Center point (nose tip or face center)
    center_idx = 30
    center_0 = face_0[center_idx]
    center_ref = face_ref[center_idx]

    # Calculate distance to center point
    dist = np.linalg.norm(face_0 - center_0, axis=1)
    dist_ref = np.linalg.norm(face_ref - center_ref, axis=1)

    # Avoid the 0 distance of the center point itself
    dist = np.delete(dist, center_idx)
    dist_ref = np.delete(dist_ref, center_idx)

    mean_dist = np.mean(dist)
    mean_dist_ref = np.mean(dist_ref)

    if mean_dist < 1e-6:
        scale_n = 1.0
    else:
        scale_n = mean_dist_ref / mean_dist

    # Clamp to [0.8, 1.5]
    scale_n = np.clip(scale_n, 0.8, 1.5)

    for i, pose in enumerate(poses):
        face = pose['faces']
        # Extract numpy array
        face = np.array(face[0])      # (68, 2)
        center = face[center_idx]
        scaled_face = (face - center) * scale_n + center
        poses[i]['faces'][0] = scaled_face

        body = pose['bodies']
        candidate = body['candidate']
        candidate_np = np.array(candidate[0])   # (14, 2)
        body_center = candidate_np[0]
        scaled_candidate = (candidate_np - body_center) * scale_n + body_center
        poses[i]['bodies']['candidate'][0] = scaled_candidate

    # In-place modification
    pose['faces'][0] = scaled_face

    return scale_n

class PoseDetectionVitPoseToDWPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": ("POSEMODEL",),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "ViTPose to DWPose format pose detection node."

    def process(self, vitpose_model, images):

        detector = vitpose_model["yolo"]
        pose_model = vitpose_model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        comfy_pbar = ProgressBar(B*2)
        progress = 0
        bboxes = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting bboxes"):
            bboxes.append(detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        kp2ds = []
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting keypoints"):
            if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])

            bbox_xywh = bbox
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        kp2ds = np.concatenate(kp2ds, 0)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)
        dwposes = [aaposemeta_to_dwpose_scail(meta) for meta in pose_metas]
        swap_hands = True
        out_dict = {"poses": dwposes, "swap_hands": swap_hands}
        return out_dict,


class ConvertOpenPoseKeypointsToDWPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keypoints": ("POSE_KEYPOINT",),
                "max_people": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1, "tooltip": "Maximum number of people to process per frame"}),
            },
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Convert OpenPose format keypoints to DWPose format."

    def process(self, keypoints, max_people=2):
        swap_hands = False
        out_dict = {"poses": convert_openpose_to_target_format(keypoints, max_people=max_people), "swap_hands": swap_hands}
        return out_dict,

class RenderDWPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 1}),
                "pose_source": (["dw_poses", "ref_dw_pose"], {"default": "dw_poses", "tooltip": "Choose which optional pose input to render"}),
                "draw_body": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw body keypoints"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw face keypoints"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw hand keypoints"}),
            },
            "optional": {
                "dw_poses": ("DWPOSES", {"default": None, "tooltip": "DW poses to render"}),
                "ref_dw_pose": ("DWPOSES", {"default": None, "tooltip": "Reference DW poses to render"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "render"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Render DW poses (or reference DW poses) to an image sequence."

    def render(self, width, height, pose_source="dw_poses", draw_body=True, draw_face=True, draw_hands=True, dw_poses=None, ref_dw_pose=None):
        selected = dw_poses if pose_source == "dw_poses" else ref_dw_pose
        # Graceful fallback if selected input is not connected.
        if selected is None:
            selected = ref_dw_pose if pose_source == "dw_poses" else dw_poses
        if selected is None:
            raise ValueError("RenderDWPose requires at least one connected input: dw_poses or ref_dw_pose.")

        pose_list = copy.deepcopy(selected["poses"])
        frames_np = draw_pose_to_canvas_np(
            pose_list,
            pool=None,
            H=height,
            W=width,
            reshape_scale=0,
            show_feet_flag=False,
            show_body_flag=draw_body,
            show_hand_flag=draw_hands,
            show_face_flag=draw_face,
            show_cheek_flag=False,
            dw_hand=True,
        )

        frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous().float() / 255.0
        mask = (frames_tensor.sum(dim=-1) > 0.0).float()

        return (frames_tensor.cpu(), mask.cpu())


class RenderDWPoseWithCameraInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 1}),
                "camera_info_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional camera info JSON from preview nodes. If enabled, frame_width/frame_height override width/height.",
                }),
                "use_camera_frame_size": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use frame_width/frame_height from camera_info_json when present.",
                }),
                "pose_source": (["dw_poses", "ref_dw_pose"], {"default": "dw_poses", "tooltip": "Choose which optional pose input to render"}),
                "draw_body": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw body keypoints"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw face keypoints"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw hand keypoints"}),
            },
            "optional": {
                "dw_poses": ("DWPOSES", {"default": None, "tooltip": "DW poses to render"}),
                "ref_dw_pose": ("DWPOSES", {"default": None, "tooltip": "Reference DW poses to render"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "render_with_camera"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Render DW poses while accepting camera_info_json input. Optionally uses camera frame size for output dimensions."

    def _extract_camera_frame_size(self, camera_info_json):
        camera_json_text = str(camera_info_json).strip() if camera_info_json is not None else ""
        if camera_json_text == "":
            return None
        try:
            data = json.loads(camera_json_text)
        except Exception as e:
            logging.warning(f"RenderDWPoseWithCameraInfo: invalid camera_info_json, ignoring frame size. Details: {e}")
            return None

        if not isinstance(data, dict):
            return None

        frame_w = data.get("frame_width")
        frame_h = data.get("frame_height", data.get("frame_length"))
        try:
            frame_w = float(frame_w)
            frame_h = float(frame_h)
        except Exception:
            return None

        if not np.isfinite(frame_w) or not np.isfinite(frame_h):
            return None
        if frame_w <= 0.0 or frame_h <= 0.0:
            return None

        # Keep dimensions within this node's declared input bounds.
        frame_w_i = int(np.clip(round(frame_w), 64, 8192))
        frame_h_i = int(np.clip(round(frame_h), 64, 8192))
        return frame_w_i, frame_h_i

    def render_with_camera(self, width, height, camera_info_json, use_camera_frame_size=True, pose_source="dw_poses", draw_body=True, draw_face=True, draw_hands=True, dw_poses=None, ref_dw_pose=None):
        render_width = int(width)
        render_height = int(height)
        if use_camera_frame_size:
            camera_size = self._extract_camera_frame_size(camera_info_json)
            if camera_size is not None:
                render_width, render_height = camera_size

        selected = dw_poses if pose_source == "dw_poses" else ref_dw_pose
        # Graceful fallback if selected input is not connected.
        if selected is None:
            selected = ref_dw_pose if pose_source == "dw_poses" else dw_poses
        if selected is None:
            raise ValueError("RenderDWPoseWithCameraInfo requires at least one connected input: dw_poses or ref_dw_pose.")

        pose_list = copy.deepcopy(selected["poses"])
        frames_np = draw_pose_to_canvas_np(
            pose_list,
            pool=None,
            H=render_height,
            W=render_width,
            reshape_scale=0,
            show_feet_flag=False,
            show_body_flag=draw_body,
            show_hand_flag=draw_hands,
            show_face_flag=draw_face,
            show_cheek_flag=False,
            dw_hand=True,
        )

        frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous().float() / 255.0
        mask = (frames_tensor.sum(dim=-1) > 0.0).float()

        return (frames_tensor.cpu(), mask.cpu())


class RenderNLFPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "nlf_poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "width": ("INT", {"default": 512}),
            "height": ("INT", {"default": 512}),
            },
            "optional": {
                "dw_poses": ("DWPOSES", {"default": None, "tooltip": "Optional DW pose model for 2D drawing"}),
                "ref_dw_pose": ("DWPOSES", {"default": None, "tooltip": "Optional reference DW pose model for alignment"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw face keypoints"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw hand keypoints"}),
                "render_device": (["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"], {"default": "gpu", "tooltip": "Taichi device to use for rendering"}),
                "scale_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to scale hand keypoints when aligning DW poses"}),
                "render_backend": (["taichi", "torch"], {"default": "taichi", "tooltip": "Rendering backend to use"}),
            }
    }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "predict"
    CATEGORY = "WanVideoWrapper"

    def _build_preview_fov_intrinsic(self, width, height, fov_degrees):
        fov = float(fov_degrees)
        if not np.isfinite(fov) or fov <= 0.0:
            fov = 55.0
        fov_rad = fov * np.pi / 180.0
        focal = float(height) / (2.0 * np.tan(fov_rad / 2.0))
        return np.array([
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def _predict_core(
        self,
        nlf_poses,
        width,
        height,
        dw_poses=None,
        ref_dw_pose=None,
        draw_face=True,
        draw_hands=True,
        render_device="gpu",
        scale_hands=True,
        render_backend="taichi",
        render_fov_degrees=55.0,
        use_preview_camera_intrinsics=False,
    ):

        from .NLFPoseExtract.nlf_render import render_nlf_as_images, render_multi_nlf_as_images, shift_dwpose_according_to_nlf, process_data_to_COCO_format, intrinsic_matrix_from_field_of_view
        from .NLFPoseExtract.align3d import solve_new_camera_params_central, solve_new_camera_params_down
        if render_backend == "taichi":
            try:
                import taichi as ti
                device_map = {
                    "cpu": ti.cpu,
                    "gpu": ti.gpu,
                    "opengl": ti.opengl,
                    "cuda": ti.cuda,
                    "vulkan": ti.vulkan,
                    "metal": ti.metal,
                }
                ti.init(arch=device_map.get(render_device.lower()))
            except:
                logging.warning("Taichi selected but not installed. Falling back to torch rendering.")
                render_backend = "torch"

        if isinstance(nlf_poses, dict):
            pose_input = nlf_poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses else nlf_poses
        else:
            pose_input = nlf_poses

        dw_pose_input = copy.deepcopy(dw_poses["poses"]) if dw_poses is not None else None
        swap_hands = dw_poses.get("swap_hands", False) if dw_poses is not None else False

        if use_preview_camera_intrinsics:
            ori_camera_pose = self._build_preview_fov_intrinsic(width, height, render_fov_degrees)
        else:
            ori_camera_pose = intrinsic_matrix_from_field_of_view([height, width], fov_degrees=render_fov_degrees)
        ori_focal = ori_camera_pose[0, 0]

        num_people = dw_pose_input[0]['bodies']['candidate'].shape[0] if dw_poses is not None else 0

        if dw_poses is not None and ref_dw_pose is not None and num_people == 1:
            ref_dw_pose_input = copy.deepcopy(ref_dw_pose["poses"])

            # Find the first valid pose
            pose_3d_first_driving_frame = None
            for pose in pose_input:
                if pose.shape[0] == 0:
                    continue
                candidate = pose[0].cpu().numpy()
                if np.any(candidate):
                    pose_3d_first_driving_frame = candidate
                    break
            if pose_3d_first_driving_frame is None:
                raise ValueError("No valid pose found in pose_input.")

            pose_3d_coco_first_driving_frame = process_data_to_COCO_format(pose_3d_first_driving_frame)
            poses_2d_ref = ref_dw_pose_input[0]['bodies']['candidate'][0][:14]
            poses_2d_ref[:, 0] = poses_2d_ref[:, 0] * width
            poses_2d_ref[:, 1] = poses_2d_ref[:, 1] * height

            poses_2d_subset = ref_dw_pose_input[0]['bodies']['subset'][0][:14]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[:14]

            valid_indices, valid_upper_indices, valid_lower_indices = [], [], []
            upper_body_indices = [0, 2, 3, 5, 6]
            lower_body_indices = [9, 10, 12, 13]

            for i in range(len(poses_2d_subset)):
                if poses_2d_subset[i] != -1.0 and np.sum(pose_3d_coco_first_driving_frame[i]) != 0:
                    if i in upper_body_indices:
                        valid_upper_indices.append(i)
                    if i in lower_body_indices:
                        valid_lower_indices.append(i)

            valid_indices = [1] + valid_lower_indices if len(valid_upper_indices) < 4 else [1] + valid_lower_indices + valid_upper_indices # align body or only lower body

            pose_2d_ref = poses_2d_ref[valid_indices]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[valid_indices]

            if len(valid_lower_indices) >= 4:
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_down(pose_3d_coco_first_driving_frame, ori_focal, [height, width], pose_2d_ref)
            else:
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_central(pose_3d_coco_first_driving_frame, ori_focal, [height, width], pose_2d_ref)

            scale_face = scale_faces(list(dw_pose_input), list(ref_dw_pose_input))   # poses[0]['faces'].shape: 1, 68, 2  , poses_ref[0]['faces'].shape: 1, 68, 2

            logging.info(f"Scale - m: {scale_m}, face: {scale_face}")
            shift_dwpose_according_to_nlf(pose_input, dw_pose_input, ori_camera_pose, new_camera_intrinsics, height, width, swap_hands=swap_hands, scale_hands=scale_hands, scale_x=scale_m, scale_y=scale_m*scale_s)

            intrinsic_matrix = new_camera_intrinsics
        else:
            intrinsic_matrix = ori_camera_pose

        if pose_input[0].shape[0] > 1:
            frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=draw_face, draw_hands=draw_hands, render_backend = render_backend)
        else:
            frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=draw_face, draw_hands=draw_hands, render_backend = render_backend)

        frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
        frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5

        return (frames_tensor.cpu().float(), mask.cpu().float())

    def predict(self, nlf_poses, width, height, dw_poses=None, ref_dw_pose=None, draw_face=True, draw_hands=True, render_device="gpu", scale_hands=True, render_backend="taichi"):
        return self._predict_core(
            nlf_poses,
            width,
            height,
            dw_poses=dw_poses,
            ref_dw_pose=ref_dw_pose,
            draw_face=draw_face,
            draw_hands=draw_hands,
            render_device=render_device,
            scale_hands=scale_hands,
            render_backend=render_backend,
            render_fov_degrees=55.0,
            use_preview_camera_intrinsics=False,
        )


class RenderNLFPosesWithCameraInfo(RenderNLFPoses):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "nlf_poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "width": ("INT", {"default": 512}),
            "height": ("INT", {"default": 512}),
            "camera_info_json": ("STRING", {"default": "", "multiline": True, "tooltip": "Camera info JSON from preview nodes. Uses fov_degrees for projection."}),
            },
            "optional": {
                "dw_poses": ("DWPOSES", {"default": None, "tooltip": "Optional DW pose model for 2D drawing"}),
                "ref_dw_pose": ("DWPOSES", {"default": None, "tooltip": "Optional reference DW pose model for alignment"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw face keypoints"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw hand keypoints"}),
                "render_device": (["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"], {"default": "gpu", "tooltip": "Taichi device to use for rendering"}),
                "scale_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to scale hand keypoints when aligning DW poses"}),
                "render_backend": (["taichi", "torch"], {"default": "taichi", "tooltip": "Rendering backend to use"}),
            }
    }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "predict_with_camera"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Render NLF poses using camera_info_json (fov_degrees) from preview nodes, without changing legacy RenderNLFPoses behavior."

    def predict_with_camera(self, nlf_poses, width, height, camera_info_json, dw_poses=None, ref_dw_pose=None, draw_face=True, draw_hands=True, render_device="gpu", scale_hands=True, render_backend="taichi"):
        render_fov_degrees = 55.0
        camera_json_text = str(camera_info_json).strip() if camera_info_json is not None else ""
        if camera_json_text != "":
            try:
                camera_data = json.loads(camera_json_text)
                if isinstance(camera_data, dict) and "fov_degrees" in camera_data:
                    render_fov_degrees = float(camera_data["fov_degrees"])
            except Exception as e:
                logging.warning(f"RenderNLFPosesWithCameraInfo: invalid camera_info_json, falling back to default fov=55.0. Details: {e}")
        if not np.isfinite(render_fov_degrees) or render_fov_degrees <= 0.0:
            render_fov_degrees = 55.0

        return self._predict_core(
            nlf_poses,
            width,
            height,
            dw_poses=dw_poses,
            ref_dw_pose=ref_dw_pose,
            draw_face=draw_face,
            draw_hands=draw_hands,
            render_device=render_device,
            scale_hands=scale_hands,
            render_backend=render_backend,
            render_fov_degrees=render_fov_degrees,
            use_preview_camera_intrinsics=True,
        )

class SaveNLFPosesAs3D:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "nlf_poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "filename_prefix": ("STRING", {"default": "nlf_pose_3d"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 300.0, "step": 0.1, "tooltip": "Frames per second for the output animation"}),
            "cylinder_radius": ("FLOAT", {"default": 21.5, "tooltip": "Radius of the cylinders representing bones"}),
            },
    }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True
    FUNCTION = "save_3d"
    CATEGORY = "WanVideoWrapper"

    def save_3d(self, nlf_poses, filename_prefix, fps, cylinder_radius):
        from .NLFPoseExtract.nlf_render import get_cylinder_specs_list_from_poses
        from .render_3d.export_utils import save_cylinder_specs_as_glb_animation
        try:
            if isinstance(nlf_poses, dict):
                pose_input = nlf_poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses else nlf_poses
            else:
                pose_input = nlf_poses

            cylinder_specs_list = get_cylinder_specs_list_from_poses(pose_input, include_missing=True)
            logging.info(f"Generated {len(cylinder_specs_list)} frames of cylinder specs")

            output_dir = folder_paths.get_output_directory()
            full_output_folder = os.path.join(output_dir, filename_prefix)
            if not os.path.exists(full_output_folder):
                os.makedirs(full_output_folder)

            filename = f"{filename_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"
            filepath = os.path.join(full_output_folder, filename)

            logging.info(f"Saving as GLB animation to {full_output_folder}")
            logging.info(f"Starting GLB animation export. Frames: {len(cylinder_specs_list)}")
            save_cylinder_specs_as_glb_animation(cylinder_specs_list, filepath, fps=fps, radius=cylinder_radius)
            logging.info(f"Saved GLB: {filepath}")
        except Exception as e:
            logging.error(f"Error in SaveNLFPosesAs3D: {e}")
            raise e

        return (filepath,)

class LoadSMPLXNPZAsNLFPred:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "smplx_model_folder": ("STRING", {"default": "", "tooltip": "Path to SMPL-X model folder that contains SMPLX_NEUTRAL.npz"}),
                "motion_npz_path": ("STRING", {"default": "", "tooltip": "Path to SMPL-X motion .npz file"}),
            },
            "optional": {
                "gender": (["auto", "neutral", "male", "female"], {"default": "auto"}),
                "num_expression_coeffs": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "joint_count": ("INT", {"default": 24, "min": 1, "max": 200, "step": 1, "tooltip": "Number of leading joints to keep for NLF-compatible output"}),
                "include_hands_face_joints": ("BOOLEAN", {"default": False, "tooltip": "If enabled, output raw SMPL-X joints (hands + face-related joints) instead of only NLF-24 joints."}),
                "per_batch": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
                "apply_translation": ("BOOLEAN", {"default": True, "tooltip": "Apply motion transl to joints for global/world positions. Disable for root-local output."}),
                "frame_load_cap": ("INT", {"default": -1, "min": -1, "max": 1000000, "step": 1, "tooltip": "Maximum frames to load after FPS resampling. -1 means no cap."}),
                "force_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.1, "tooltip": "If >0, resample sequence to this FPS before FK."}),
            },
        }

    RETURN_TYPES = ("NLFPRED", "NLFPRED",)
    RETURN_NAMES = ("pose_results_24", "pose_results_all",)
    FUNCTION = "load_and_fk"
    CATEGORY = "SCAIL-Pose"
    DESCRIPTION = "Load a SMPL-X motion npz, run FK with SMPL-X forward(), and output two NLFPRED streams: 24-joint NLF-compatible output plus an all-joints output (raw SMPL-X joints when include_hands_face_joints is enabled)."

    def _as_float_tensor(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.float()
        return torch.tensor(value, dtype=torch.float32)

    def _npz_get(self, motion, keys):
        for key in keys:
            if key in motion:
                return motion[key]
        return None

    def _normalize_pose_component(self, value, expected_dim):
        if value is None:
            return None
        t = self._as_float_tensor(value)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        if t.ndim == 3:
            t = t.reshape(t.shape[0], -1)
        if t.shape[-1] != expected_dim:
            raise ValueError(f"Expected last dim {expected_dim}, got {t.shape[-1]}.")
        return t

    def _ensure_frame_count(self, tensor_value, n_frames, name):
        if tensor_value is None:
            return None
        if tensor_value.shape[0] == n_frames:
            return tensor_value
        if tensor_value.shape[0] == 1 and n_frames > 1:
            reps = [n_frames] + [1] * (tensor_value.ndim - 1)
            return tensor_value.repeat(*reps)
        raise ValueError(f"{name} has {tensor_value.shape[0]} frames, expected {n_frames} (or 1 for broadcasting).")

    def _decode_gender(self, raw_gender):
        if raw_gender is None:
            return "neutral"
        if isinstance(raw_gender, np.ndarray):
            if raw_gender.ndim == 0:
                raw_gender = raw_gender.item()
            elif raw_gender.size > 0:
                raw_gender = raw_gender.reshape(-1)[0]
        if isinstance(raw_gender, bytes):
            raw_gender = raw_gender.decode("utf-8", errors="ignore")
        raw = str(raw_gender).strip().lower()
        if raw in ("male", "female", "neutral"):
            return raw
        return "neutral"

    def _decode_fps(self, raw_value):
        if raw_value is None:
            return None
        if isinstance(raw_value, np.ndarray):
            if raw_value.ndim == 0:
                raw_value = raw_value.item()
            elif raw_value.size > 0:
                raw_value = raw_value.reshape(-1)[0]
            else:
                return None
        if torch.is_tensor(raw_value):
            if raw_value.numel() == 0:
                return None
            raw_value = raw_value.reshape(-1)[0].item()
        try:
            fps = float(raw_value)
        except Exception:
            return None
        if not np.isfinite(fps) or fps <= 0:
            return None
        return fps

    def _select_frames(self, tensor_value, frame_indices, original_n_frames, name):
        if tensor_value is None:
            return None
        if tensor_value.shape[0] == original_n_frames:
            return tensor_value.index_select(0, frame_indices)
        if tensor_value.shape[0] == 1:
            reps = [frame_indices.shape[0]] + [1] * (tensor_value.ndim - 1)
            return tensor_value.repeat(*reps)
        raise ValueError(
            f"{name} has {tensor_value.shape[0]} frames, expected {original_n_frames} (or 1 for broadcasting)."
        )

    def _smplx_to_nlf_24_joints(self, joints_full):
        # Build SMPL-style 24 joints from SMPL-X output.
        # SMPL-X indices 0..21 match pelvis..wrists. SMPL indices 22/23 are left/right hand,
        # which are approximated from finger base joints in SMPL-X.
        if joints_full.shape[1] < 53:
            raise ValueError(f"SMPL-X joints output too short: {joints_full.shape}")

        nlf_joints = torch.zeros((joints_full.shape[0], 24, 3), dtype=joints_full.dtype, device=joints_full.device)
        nlf_joints[:, :22, :] = joints_full[:, :22, :]

        # Finger base joints (index/middle/pinky/ring/thumb) for left and right hands.
        left_hand_src = [25, 28, 31, 34, 37]
        right_hand_src = [40, 43, 46, 49, 52]

        nlf_joints[:, 22, :] = joints_full[:, left_hand_src, :].mean(dim=1)
        nlf_joints[:, 23, :] = joints_full[:, right_hand_src, :].mean(dim=1)
        return nlf_joints

    def load_and_fk(self, smplx_model_folder, motion_npz_path, gender="auto", num_expression_coeffs=10, joint_count=24, include_hands_face_joints=False, include_face_contour=True, per_batch=64, apply_translation=True, frame_load_cap=-1, force_fps=0.0):
        if not smplx_model_folder or not os.path.exists(smplx_model_folder):
            raise ValueError(f"Invalid SMPL-X model folder: {smplx_model_folder}")
        if not motion_npz_path or not os.path.isfile(motion_npz_path):
            raise ValueError(f"Invalid motion npz path: {motion_npz_path}")

        try:
            import smplx
        except Exception as e:
            raise ImportError(f"Failed to import installed smplx package. Please install it first. Details: {e}")

        motion_file = np.load(motion_npz_path, allow_pickle=True)
        motion = {k: motion_file[k] for k in motion_file.files}

        motion_gender = self._decode_gender(self._npz_get(motion, ["gender"]))
        selected_gender = motion_gender if gender == "auto" else gender

        poses = self._npz_get(motion, ["poses", "smpl_poses"])
        poses_t = None
        if poses is not None:
            poses_t = self._as_float_tensor(poses)
            if poses_t.ndim == 3:
                poses_t = poses_t.reshape(poses_t.shape[0], -1)
            if poses_t.ndim != 2 or poses_t.shape[-1] < 66:
                raise ValueError(f"Unsupported poses shape: {tuple(poses_t.shape)}")

        global_orient = self._normalize_pose_component(self._npz_get(motion, ["global_orient", "root_orient"]), 3)
        body_pose = self._normalize_pose_component(self._npz_get(motion, ["body_pose", "pose_body"]), 63)
        left_hand_pose = self._normalize_pose_component(self._npz_get(motion, ["left_hand_pose"]), 45)
        right_hand_pose = self._normalize_pose_component(self._npz_get(motion, ["right_hand_pose"]), 45)
        jaw_pose = self._normalize_pose_component(self._npz_get(motion, ["jaw_pose"]), 3)
        leye_pose = self._normalize_pose_component(self._npz_get(motion, ["leye_pose", "left_eye_pose"]), 3)
        reye_pose = self._normalize_pose_component(self._npz_get(motion, ["reye_pose", "right_eye_pose"]), 3)
        expression = self._as_float_tensor(self._npz_get(motion, ["expression", "expressions"]))
        transl = self._as_float_tensor(self._npz_get(motion, ["transl", "trans", "translation"]))

        if expression is not None:
            if expression.ndim == 1:
                expression = expression.unsqueeze(0)
            if expression.ndim == 3:
                expression = expression.reshape(expression.shape[0], -1)

        if transl is not None:
            if transl.ndim == 1:
                transl = transl.unsqueeze(0)
            if transl.ndim == 3:
                transl = transl.reshape(transl.shape[0], -1)

        pose_hand = self._as_float_tensor(self._npz_get(motion, ["pose_hand"]))
        if pose_hand is not None:
            if pose_hand.ndim == 3:
                pose_hand = pose_hand.reshape(pose_hand.shape[0], -1)
            if pose_hand.shape[-1] >= 90:
                if left_hand_pose is None:
                    left_hand_pose = pose_hand[:, :45]
                if right_hand_pose is None:
                    right_hand_pose = pose_hand[:, 45:90]

        if poses_t is not None:
            if global_orient is None:
                global_orient = poses_t[:, :3]
            if body_pose is None and poses_t.shape[-1] >= 66:
                body_pose = poses_t[:, 3:66]
            if jaw_pose is None and poses_t.shape[-1] >= 69:
                jaw_pose = poses_t[:, 66:69]
            if leye_pose is None and poses_t.shape[-1] >= 72:
                leye_pose = poses_t[:, 69:72]
            if reye_pose is None and poses_t.shape[-1] >= 75:
                reye_pose = poses_t[:, 72:75]
            if left_hand_pose is None and poses_t.shape[-1] >= 120:
                left_hand_pose = poses_t[:, 75:120]
            if right_hand_pose is None and poses_t.shape[-1] >= 165:
                right_hand_pose = poses_t[:, 120:165]

        n_frames = None
        for tensor_candidate in [global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, transl, expression, poses_t]:
            if tensor_candidate is not None:
                n_frames = tensor_candidate.shape[0]
                break
        if n_frames is None:
            raise ValueError("Motion npz does not contain usable SMPL-X pose parameters.")

        original_n_frames = n_frames
        source_fps = self._decode_fps(self._npz_get(motion, ["fps", "source_fps", "mocap_framerate", "mocap_frame_rate", "framerate", "frame_rate"]))
        if source_fps is None:
            source_fps = 30.0

        if force_fps is not None and force_fps > 0:
            target_len = max(1, int(round(original_n_frames * float(force_fps) / float(source_fps))))
            target_times = torch.arange(target_len, dtype=torch.float32) / float(force_fps)
            frame_indices = torch.clamp(torch.round(target_times * float(source_fps)).to(torch.long), 0, original_n_frames - 1)
        else:
            frame_indices = torch.arange(original_n_frames, dtype=torch.long)

        if frame_load_cap is not None and frame_load_cap > 0:
            frame_indices = frame_indices[:int(frame_load_cap)]

        if frame_indices.numel() == 0:
            raise ValueError("No frames selected. Increase frame_load_cap or disable force_fps.")

        n_frames = int(frame_indices.shape[0])
        if force_fps is not None and force_fps > 0:
            logging.info(f"Resampled motion FPS from {source_fps:.3f} to {float(force_fps):.3f}; frames {original_n_frames}->{n_frames}")
        elif frame_load_cap is not None and frame_load_cap > 0 and n_frames < original_n_frames:
            logging.info(f"Applied frame cap: frames {original_n_frames}->{n_frames}")

        global_orient = self._select_frames(global_orient, frame_indices, original_n_frames, "global_orient")
        body_pose = self._select_frames(body_pose, frame_indices, original_n_frames, "body_pose")
        left_hand_pose = self._select_frames(left_hand_pose, frame_indices, original_n_frames, "left_hand_pose")
        right_hand_pose = self._select_frames(right_hand_pose, frame_indices, original_n_frames, "right_hand_pose")
        jaw_pose = self._select_frames(jaw_pose, frame_indices, original_n_frames, "jaw_pose")
        leye_pose = self._select_frames(leye_pose, frame_indices, original_n_frames, "leye_pose")
        reye_pose = self._select_frames(reye_pose, frame_indices, original_n_frames, "reye_pose")
        transl = self._select_frames(transl, frame_indices, original_n_frames, "transl")
        expression = self._select_frames(expression, frame_indices, original_n_frames, "expression")

        if global_orient is None:
            global_orient = torch.zeros((n_frames, 3), dtype=torch.float32)
        if body_pose is None:
            body_pose = torch.zeros((n_frames, 63), dtype=torch.float32)
        if left_hand_pose is None:
            left_hand_pose = torch.zeros((n_frames, 45), dtype=torch.float32)
        if right_hand_pose is None:
            right_hand_pose = torch.zeros((n_frames, 45), dtype=torch.float32)
        if jaw_pose is None:
            jaw_pose = torch.zeros((n_frames, 3), dtype=torch.float32)
        if leye_pose is None:
            leye_pose = torch.zeros((n_frames, 3), dtype=torch.float32)
        if reye_pose is None:
            reye_pose = torch.zeros((n_frames, 3), dtype=torch.float32)
        if transl is None:
            transl = torch.zeros((n_frames, 3), dtype=torch.float32)

        global_orient = self._ensure_frame_count(global_orient, n_frames, "global_orient")
        body_pose = self._ensure_frame_count(body_pose, n_frames, "body_pose")
        left_hand_pose = self._ensure_frame_count(left_hand_pose, n_frames, "left_hand_pose")
        right_hand_pose = self._ensure_frame_count(right_hand_pose, n_frames, "right_hand_pose")
        jaw_pose = self._ensure_frame_count(jaw_pose, n_frames, "jaw_pose")
        leye_pose = self._ensure_frame_count(leye_pose, n_frames, "leye_pose")
        reye_pose = self._ensure_frame_count(reye_pose, n_frames, "reye_pose")
        transl = self._ensure_frame_count(transl, n_frames, "transl")

        betas_raw = self._as_float_tensor(self._npz_get(motion, ["betas"]))
        if betas_raw is None:
            betas_raw = torch.zeros((10,), dtype=torch.float32)
        if betas_raw.ndim > 1:
            betas_raw = betas_raw.reshape(-1, betas_raw.shape[-1])[0]
        betas = betas_raw.unsqueeze(0)

        # Face contour is always enabled when requesting raw SMPL-X joints.
        # Keep include_face_contour arg for backward compatibility with older workflows.
        use_face_contour = bool(include_hands_face_joints)

        model = smplx.create(
            smplx_model_folder,
            model_type="smplx",
            gender=selected_gender,
            use_face_contour=use_face_contour,
            num_betas=betas.shape[-1],
            num_expression_coeffs=num_expression_coeffs,
            use_pca=False,
            ext="npz",
        )

        device_local = torch.device("cpu")
        model = model.to(device_local).eval()

        global_orient = global_orient.to(device_local)
        body_pose = body_pose.to(device_local)
        left_hand_pose = left_hand_pose.to(device_local)
        right_hand_pose = right_hand_pose.to(device_local)
        jaw_pose = jaw_pose.to(device_local)
        leye_pose = leye_pose.to(device_local)
        reye_pose = reye_pose.to(device_local)
        transl = transl.to(device_local)
        betas = betas.to(device_local)

        if expression is not None:
            expression = expression.to(device_local)
            if expression.ndim == 1:
                expression = expression.unsqueeze(0)
            if expression.ndim == 3:
                expression = expression.reshape(expression.shape[0], -1)
            if expression.shape[-1] > num_expression_coeffs:
                expression = expression[:, :num_expression_coeffs]
            elif expression.shape[-1] < num_expression_coeffs:
                pad = torch.zeros((expression.shape[0], num_expression_coeffs - expression.shape[-1]), dtype=expression.dtype, device=expression.device)
                expression = torch.cat([expression, pad], dim=-1)
        else:
            expression = torch.zeros((n_frames, num_expression_coeffs), dtype=torch.float32, device=device_local)

        expression = self._ensure_frame_count(expression, n_frames, "expression")

        if betas.shape[-1] > model.num_betas:
            betas = betas[:, :model.num_betas]
        elif betas.shape[-1] < model.num_betas:
            pad = torch.zeros((1, model.num_betas - betas.shape[-1]), dtype=betas.dtype, device=betas.device)
            betas = torch.cat([betas, pad], dim=-1)

        all_joints3d_24 = []
        all_joints3d_all = []
        with torch.no_grad():
            for i in range(0, n_frames, per_batch):
                end = min(i + per_batch, n_frames)
                current_batch = end - i
                # Keep FK call close to the original scripts: betas + global/body/hands.
                # Translation is optional because many motion files store world-space transl that
                # should not be applied for NLF-style camera-space rendering.
                out = model(
                    betas=betas.expand(current_batch, -1),
                    global_orient=global_orient[i:end],
                    body_pose=body_pose[i:end],
                    jaw_pose=jaw_pose[i:end],
                    leye_pose=leye_pose[i:end],
                    reye_pose=reye_pose[i:end],
                    expression=expression[i:end],
                    left_hand_pose=left_hand_pose[i:end],
                    right_hand_pose=right_hand_pose[i:end],
                    transl=transl[i:end] if apply_translation else None,
                    return_verts=False,
                )

                joints_24 = self._smplx_to_nlf_24_joints(out.joints) * 1000.0
                joints_all = out.joints * 1000.0

                # Output-1: NLF-compatible 24 joints (or fewer when joint_count < 24).
                keep_count_24 = min(int(joint_count), int(joints_24.shape[1]))
                out_24 = joints_24[:, :keep_count_24, :].to(offload_device)

                # Output-2: all joints stream. Raw SMPL-X when enabled, else mirrors output-1.
                if include_hands_face_joints:
                    out_all = joints_all.to(offload_device)
                else:
                    out_all = out_24

                for frame_idx in range(out_24.shape[0]):
                    all_joints3d_24.append(out_24[frame_idx:frame_idx + 1])
                    all_joints3d_all.append(out_all[frame_idx:frame_idx + 1])

        pose_results_24 = {
            "joints3d_nonparam": [all_joints3d_24],
        }
        pose_results_all = {
            "joints3d_nonparam": [all_joints3d_all],
        }

        joints_per_frame_24 = int(all_joints3d_24[0].shape[1]) if len(all_joints3d_24) > 0 else 0
        joints_per_frame_all = int(all_joints3d_all[0].shape[1]) if len(all_joints3d_all) > 0 else 0
        layout_name = "smplx_raw" if include_hands_face_joints else "nlf24-mirrored"
        logging.info(f"Loaded SMPL-X motion: {n_frames} frames, output24_joints/frame={joints_per_frame_24}, outputAll_joints/frame={joints_per_frame_all}, all_layout={layout_name}, gender={selected_gender}, face_contour={use_face_contour}")
        return (pose_results_24, pose_results_all,)

class ConvertWorldNLFPoseToCameraSpace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_poses_world": ("NLFPRED", {"tooltip": "World-space NLF poses"}),
            },
            "optional": {
                "camera_info_json": ("STRING", {"default": "", "tooltip": "Optional camera info JSON. Supports world_to_camera (3x4/4x4), camera_to_world (4x4), or {R,t}/{camera_position,camera_rotation_xyz_deg}."}),
                "camera_info_source": (["auto", "generic", "threejs"], {"default": "auto", "tooltip": "Interpret camera_info_json extrinsics convention. Use threejs for camera matrices captured from Three.js camera.matrixWorld (right-handed, -Z forward)."}),
                "cam_position_x": ("FLOAT", {"default": 0.0, "min": -100000.0, "max": 100000.0, "step": 0.01, "tooltip": "Default camera world X if camera_info_json is empty"}),
                "cam_position_y": ("FLOAT", {"default": -800.0, "min": -100000.0, "max": 100000.0, "step": 0.01, "tooltip": "Default camera world Y if camera_info_json is empty"}),
                "cam_position_z": ("FLOAT", {"default": -4000.0, "min": -100000.0, "max": 100000.0, "step": 0.01, "tooltip": "Default camera world Z if camera_info_json is empty (more negative moves camera farther back)"}),
                "cam_rotation_x_deg": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.1, "tooltip": "Default camera rotation X (degrees) if camera_info_json is empty"}),
                "cam_rotation_y_deg": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.1, "tooltip": "Default camera rotation Y (degrees) if camera_info_json is empty"}),
                "cam_rotation_z_deg": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.1, "tooltip": "Default camera rotation Z (degrees) if camera_info_json is empty"}),
                "camera_axis_correction": (["none", "flip_y", "flip_yz", "Zup->Yup"], {"default": "Zup->Yup", "tooltip": "World-space axis correction applied before world->camera."}),
            },
        }

    RETURN_TYPES = ("NLFPRED",)
    RETURN_NAMES = ("nlf_poses_camera",)
    FUNCTION = "convert"
    CATEGORY = "SCAIL-Pose"
    DESCRIPTION = "Convert world-space NLF poses into camera-space 3D using provided extrinsics or a default camera. Includes optional axis correction and Three.js camera-matrix compatibility."

    def _euler_xyz_to_matrix(self, rx_deg, ry_deg, rz_deg):
        rx = torch.tensor(rx_deg * np.pi / 180.0, dtype=torch.float32)
        ry = torch.tensor(ry_deg * np.pi / 180.0, dtype=torch.float32)
        rz = torch.tensor(rz_deg * np.pi / 180.0, dtype=torch.float32)

        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        rx_m = torch.tensor([[1.0, 0.0, 0.0], [0.0, cx.item(), -sx.item()], [0.0, sx.item(), cx.item()]], dtype=torch.float32)
        ry_m = torch.tensor([[cy.item(), 0.0, sy.item()], [0.0, 1.0, 0.0], [-sy.item(), 0.0, cy.item()]], dtype=torch.float32)
        rz_m = torch.tensor([[cz.item(), -sz.item(), 0.0], [sz.item(), cz.item(), 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

        # Apply XYZ euler order to camera orientation.
        return rz_m @ ry_m @ rx_m

    def _should_apply_threejs_camera_fix(self, camera_info_source, data=None):
        source = str(camera_info_source or "auto").strip().lower()
        if source == "threejs":
            return True
        if source == "generic":
            return False

        if isinstance(data, dict):
            for key in ("camera_info_source", "camera_source", "camera_convention", "source", "convention"):
                if key not in data:
                    continue
                value = str(data[key]).strip().lower()
                if "three" in value and "js" in value:
                    return True
                if value in ("generic", "opencv", "cv"):
                    return False

            # Heuristic for JSON emitted by the in-node Three.js preview widget.
            if "camera_to_world" in data and any(k in data for k in ("target", "aspect", "fov_degrees", "near", "far")):
                return True

        return False

    def _apply_threejs_camera_fix(self, r, t):
        # Three.js camera space is right-handed with -Z forward and +Y up.
        # Convert to the projection convention used here (+Z forward, +Y down).
        s = r.new_tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        return s @ r, s @ t

    def _extract_world_to_camera(self, camera_info_json, camera_info_source, cam_position_x, cam_position_y, cam_position_z, cam_rotation_x_deg, cam_rotation_y_deg, cam_rotation_z_deg):
        if camera_info_json is not None and str(camera_info_json).strip() != "":
            try:
                data = json.loads(camera_info_json)
            except Exception as e:
                raise ValueError(f"Invalid camera_info_json: {e}")

            if not isinstance(data, dict):
                raise ValueError("camera_info_json must decode to an object/dict.")

            unit_scale = 1.0
            for scale_key in ("camera_unit_scale", "unit_scale", "position_scale"):
                if scale_key in data:
                    try:
                        unit_scale = float(data[scale_key])
                    except Exception:
                        raise ValueError(f"{scale_key} must be a positive number.")
                    break
            if not np.isfinite(unit_scale) or unit_scale <= 0.0:
                raise ValueError("camera unit scale must be a positive finite number.")

            apply_threejs_fix = self._should_apply_threejs_camera_fix(camera_info_source, data)

            if "world_to_camera" in data:
                m = torch.tensor(data["world_to_camera"], dtype=torch.float32)
                if m.shape == (4, 4):
                    r, t = m[:3, :3], m[:3, 3] * unit_scale
                    if apply_threejs_fix:
                        r, t = self._apply_threejs_camera_fix(r, t)
                    return r, t
                if m.shape == (3, 4):
                    r, t = m[:, :3], m[:, 3] * unit_scale
                    if apply_threejs_fix:
                        r, t = self._apply_threejs_camera_fix(r, t)
                    return r, t
                if m.shape == (3, 3):
                    r, t = m, torch.zeros((3,), dtype=torch.float32)
                    if apply_threejs_fix:
                        r, t = self._apply_threejs_camera_fix(r, t)
                    return r, t
                raise ValueError(f"world_to_camera shape must be 3x3, 3x4, or 4x4. Got {tuple(m.shape)}")

            if "camera_to_world" in data:
                m = torch.tensor(data["camera_to_world"], dtype=torch.float32)
                if m.shape != (4, 4):
                    raise ValueError(f"camera_to_world must be 4x4. Got {tuple(m.shape)}")
                if unit_scale != 1.0:
                    m = m.clone()
                    m[:3, 3] = m[:3, 3] * unit_scale
                w2c = torch.linalg.inv(m)
                r, t = w2c[:3, :3], w2c[:3, 3]
                if apply_threejs_fix:
                    r, t = self._apply_threejs_camera_fix(r, t)
                return r, t

            if "R" in data:
                r = torch.tensor(data["R"], dtype=torch.float32)
                if r.shape != (3, 3):
                    raise ValueError(f"R must be 3x3. Got {tuple(r.shape)}")
                if "t" in data:
                    t = torch.tensor(data["t"], dtype=torch.float32).reshape(-1)
                    if t.numel() != 3:
                        raise ValueError("t must have 3 elements.")
                    if apply_threejs_fix:
                        r, t = self._apply_threejs_camera_fix(r, t)
                    return r, t * unit_scale
                c = torch.tensor(data.get("camera_position", [0.0, 0.0, 0.0]), dtype=torch.float32).reshape(-1)
                if c.numel() != 3:
                    raise ValueError("camera_position must have 3 elements.")
                c = c * unit_scale
                t = -(r @ c)
                if apply_threejs_fix:
                    r, t = self._apply_threejs_camera_fix(r, t)
                return r, t

            if "camera_position" in data or "camera_rotation_xyz_deg" in data:
                c = torch.tensor(data.get("camera_position", [0.0, 0.0, 0.0]), dtype=torch.float32).reshape(-1)
                if c.numel() != 3:
                    raise ValueError("camera_position must have 3 elements.")
                c = c * unit_scale
                rot = data.get("camera_rotation_xyz_deg", [0.0, 0.0, 0.0])
                if len(rot) != 3:
                    raise ValueError("camera_rotation_xyz_deg must have 3 elements.")
                r = self._euler_xyz_to_matrix(rot[0], rot[1], rot[2])
                t = -(r @ c)
                if apply_threejs_fix:
                    r, t = self._apply_threejs_camera_fix(r, t)
                return r, t

            raise ValueError("camera_info_json missing supported keys. Use world_to_camera, camera_to_world, {R,t}, or {camera_position,camera_rotation_xyz_deg}.")

        # Default camera when no camera info is provided.
        c = torch.tensor([cam_position_x, cam_position_y, cam_position_z], dtype=torch.float32)
        r = self._euler_xyz_to_matrix(cam_rotation_x_deg, cam_rotation_y_deg, cam_rotation_z_deg)
        t = -(r @ c)
        if self._should_apply_threejs_camera_fix(camera_info_source, None):
            r, t = self._apply_threejs_camera_fix(r, t)
        return r, t

    def convert(self, nlf_poses_world, camera_info_json="", camera_info_source="auto", cam_position_x=0.0, cam_position_y=-800.0, cam_position_z=-4000.0, cam_rotation_x_deg=0.0, cam_rotation_y_deg=0.0, cam_rotation_z_deg=0.0, camera_axis_correction="Zup->Yup"):
        if isinstance(nlf_poses_world, dict):
            pose_source = nlf_poses_world['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses_world else nlf_poses_world
        else:
            pose_source = nlf_poses_world

        if not isinstance(pose_source, (list, tuple)):
            raise ValueError("Expected nlf_poses_world to contain a list of frame tensors.")

        # Clone the incoming frames so this node never mutates upstream NLFPRED data.
        pose_input = []
        for frame in pose_source:
            if torch.is_tensor(frame):
                pose_input.append(frame.detach().clone())
            else:
                pose_input.append(copy.deepcopy(frame))

        # Keep compatibility with older workflows that may still pass "auto".
        axis_mode = "Zup->Yup" if camera_axis_correction == "auto" else camera_axis_correction

        r, t = self._extract_world_to_camera(
            camera_info_json,
            camera_info_source,
            cam_position_x, cam_position_y, cam_position_z,
            cam_rotation_x_deg, cam_rotation_y_deg, cam_rotation_z_deg,
        )

        all_joints3d = []
        for frame in pose_input:
            if torch.is_tensor(frame):
                frame_t = frame.detach().float().cpu().clone()
            else:
                frame_t = torch.tensor(frame, dtype=torch.float32)

            if frame_t.ndim == 2 and frame_t.shape[-1] == 3:
                frame_t = frame_t.unsqueeze(0)
            if frame_t.ndim != 3 or frame_t.shape[-1] != 3:
                raise ValueError(f"Expected frame shape [P,J,3] or [J,3], got {tuple(frame_t.shape)}")

            if axis_mode == "flip_y":
                frame_t[..., 1] = -frame_t[..., 1]
            elif axis_mode == "flip_yz":
                frame_t[..., 1] = -frame_t[..., 1]
                frame_t[..., 2] = -frame_t[..., 2]
            elif axis_mode == "Zup->Yup":
                z_old = frame_t[..., 2].clone()
                frame_t[..., 2] = frame_t[..., 1]
                frame_t[..., 1] = -z_old

            cam = torch.matmul(frame_t, r.t()) + t.view(1, 1, 3)
            all_joints3d.append(cam.to(offload_device))

        return ({"joints3d_nonparam": [all_joints3d]},)

class ConvertWorldNLFPoseToDWPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_poses_world": ("NLFPRED", {"tooltip": "World-space NLF poses (for example from LoadSMPLXNPZAsNLFPred)"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 1}),
            },
            "optional": {
                "camera_info_json": ("STRING", {"default": "", "tooltip": "Optional camera info JSON. Supports world_to_camera (3x4/4x4), camera_to_world (4x4), or {R,t}/{camera_position,camera_rotation_xyz_deg}."}),
                "camera_info_source": (["auto", "generic", "threejs"], {"default": "auto", "tooltip": "Interpret camera_info_json extrinsics convention. Use threejs for camera matrices captured from Three.js camera.matrixWorld."}),
                "cam_position_x": ("FLOAT", {"default": 0.0, "min": -100000.0, "max": 100000.0, "step": 0.01, "tooltip": "Default camera world X if camera_info_json is empty"}),
                "cam_position_y": ("FLOAT", {"default": -800.0, "min": -100000.0, "max": 100000.0, "step": 0.01, "tooltip": "Default camera world Y if camera_info_json is empty"}),
                "cam_position_z": ("FLOAT", {"default": -4000.0, "min": -100000.0, "max": 100000.0, "step": 0.01, "tooltip": "Default camera world Z if camera_info_json is empty (more negative moves camera farther back)"}),
                "cam_rotation_x_deg": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.1, "tooltip": "Default camera rotation X (degrees) if camera_info_json is empty"}),
                "cam_rotation_y_deg": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.1, "tooltip": "Default camera rotation Y (degrees) if camera_info_json is empty"}),
                "cam_rotation_z_deg": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.1, "tooltip": "Default camera rotation Z (degrees) if camera_info_json is empty"}),
                "camera_axis_correction": (["none", "flip_y", "flip_yz", "Zup->Yup"], {"default": "Zup->Yup", "tooltip": "World-space axis correction applied before world->camera."}),
                "fov_degrees": ("FLOAT", {"default": 55.0, "min": 1.0, "max": 179.0, "step": 0.1, "tooltip": "Field of view used to build intrinsic matrix for 3D->2D projection."}),
            },
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "convert"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Convert world-space NLF poses to DWPOSES by sharing camera settings with ConvertWorldNLFPoseToCameraSpace (including Three.js matrix support), then projecting to DW/OpenPose body order."

    def _build_preview_fov_intrinsic(self, width, height, fov_degrees):
        fov = float(fov_degrees)
        if not np.isfinite(fov) or fov <= 0.0:
            fov = 55.0
        fov_rad = fov * np.pi / 180.0
        focal = float(height) / (2.0 * np.tan(fov_rad / 2.0))
        return np.array([
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def _resolve_render_fov(self, fov_degrees, camera_info_json=""):
        resolved_fov = float(fov_degrees)
        camera_json_text = str(camera_info_json).strip() if camera_info_json is not None else ""
        if camera_json_text != "":
            try:
                camera_data = json.loads(camera_json_text)
                if isinstance(camera_data, dict) and "fov_degrees" in camera_data:
                    resolved_fov = float(camera_data["fov_degrees"])
            except Exception as e:
                logging.warning(f"ConvertWorldNLFPoseToDWPose: invalid camera_info_json for fov parsing, falling back to node fov_degrees. Details: {e}")

        if not np.isfinite(resolved_fov) or resolved_fov <= 0.0:
            resolved_fov = 55.0
        return resolved_fov

    def _project_points_3d(self, points_3d, intrinsic_matrix, width, height):
        points = np.asarray(points_3d, dtype=np.float32)
        out_xy = np.full((points.shape[0], 2), -1.0, dtype=np.float32)
        out_score = np.zeros((points.shape[0],), dtype=np.float32)
        if points.shape[0] == 0:
            return out_xy, out_score

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 1e-6)
        if np.any(valid):
            u = (intrinsic_matrix[0, 0] * x[valid] / z[valid]) + intrinsic_matrix[0, 2]
            v = (intrinsic_matrix[1, 1] * y[valid] / z[valid]) + intrinsic_matrix[1, 2]
            out_xy[valid, 0] = u / float(width)
            out_xy[valid, 1] = v / float(height)
            out_score[valid] = 1.0
        return out_xy, out_score

    def _pick_joint(self, xyz, candidates):
        for idx in candidates:
            if 0 <= idx < xyz.shape[0]:
                p = xyz[idx]
                if np.isfinite(p).all() and p[2] > 1e-6:
                    return p.astype(np.float32)
        return None

    def _get_smplx_to_coco_wholebody_src_indices(self):
        if hasattr(self, "_smplx_to_coco_wholebody_src_indices"):
            return self._smplx_to_coco_wholebody_src_indices

        # Citation:
        # Adapted from OpenMMLab mmhuman3d keypoint conventions + convert_kps logic:
        # - mmhuman3d/core/conventions/keypoints_mapping/smplx.py
        # - mmhuman3d/core/conventions/keypoints_mapping/coco_wholebody.py
        # - mmhuman3d/core/conventions/keypoints_mapping/__init__.py (convert_kps)
        # Repository: https://github.com/open-mmlab/mmhuman3d
        src_indices = np.array([
            # coco_wholebody body+feet(23)
            55, 57, 56, 59, 58, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8, 60, 61, 62, 63, 64, 65,
            # face contour(17)
            127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
            # brows+nose+eyes+mouth+lips(51)
            76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
            86, 87, 88, 89, 90, 91, 92, 93, 94,
            95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
            107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
            119, 120, 121, 122, 123, 124, 125, 126,
            # left hand (21): root + thumb/index/middle/ring/pinky
            20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70,
            # right hand (21): root + thumb/index/middle/ring/pinky
            21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75,
        ], dtype=np.int32)
        self._smplx_to_coco_wholebody_src_indices = src_indices
        return src_indices

    def _convert_kps_smplx_to_coco_wholebody(self, smplx_keypoints_144, src_mask=None):
        # Minimal local implementation of convert_kps for src='smplx', dst='coco_wholebody'.
        # See citation in _get_smplx_to_coco_wholebody_src_indices.
        kps = np.asarray(smplx_keypoints_144, dtype=np.float32)
        if kps.shape != (144, 3):
            raise ValueError(f"Expected smplx keypoints shape (144,3), got {kps.shape}")

        if src_mask is None:
            src_mask = np.ones((144,), dtype=np.uint8)
        else:
            src_mask = np.asarray(src_mask, dtype=np.uint8).reshape(-1)
            if src_mask.shape[0] != 144:
                raise ValueError(f"Expected src_mask shape (144,), got {src_mask.shape}")

        src_indices = self._get_smplx_to_coco_wholebody_src_indices()
        out_kps = np.zeros((133, 3), dtype=np.float32)
        out_mask = np.zeros((133,), dtype=np.uint8)
        out_kps[:, :] = kps[src_indices, :]
        out_mask[:] = src_mask[src_indices]
        return out_kps, out_mask

    def _to_coco_wholebody(self, xyz):
        xyz = np.asarray(xyz, dtype=np.float32)
        n_src = int(xyz.shape[0])

        # Already coco_wholebody-133.
        if n_src == 133:
            return xyz, np.ones((133,), dtype=np.uint8)

        # Need SMPL-X facial region (>=127) to map robustly to coco_wholebody.
        if n_src < 127:
            return None, None

        smplx_144 = np.zeros((144, 3), dtype=np.float32)
        src_mask = np.zeros((144,), dtype=np.uint8)
        copy_n = min(144, n_src)
        smplx_144[:copy_n, :] = xyz[:copy_n, :]
        src_mask[:copy_n] = 1

        coco_xyz, coco_mask = self._convert_kps_smplx_to_coco_wholebody(smplx_144, src_mask)
        if coco_xyz.shape[0] != 133:
            return None, None
        if coco_mask.shape[0] != 133:
            coco_mask = np.ones((133,), dtype=np.uint8)
        return coco_xyz, coco_mask

    def _project_person_to_dw_from_coco(self, xyz_orig, coco_xyz, coco_mask, intrinsic_matrix, width, height):
        out_candidate = np.full((18, 2), -1.0, dtype=np.float32)
        out_subset = np.full((18,), -1.0, dtype=np.float32)
        out_score = np.zeros((18,), dtype=np.float32)

        def _valid_coco(idx):
            if not (0 <= idx < coco_xyz.shape[0]):
                return False
            if coco_mask is not None and int(coco_mask[idx]) == 0:
                return False
            p = coco_xyz[idx]
            return np.isfinite(p).all() and p[2] > 1e-6

        # COCO-WholeBody -> OpenPose body18 order used by draw_bodypose.
        coco_to_dw = {
            0: 0,    # nose
            6: 2,    # right_shoulder
            8: 3,    # right_elbow
            10: 4,   # right_wrist
            5: 5,    # left_shoulder
            7: 6,    # left_elbow
            9: 7,    # left_wrist
            12: 8,   # right_hip
            14: 9,   # right_knee
            16: 10,  # right_ankle
            11: 11,  # left_hip
            13: 12,  # left_knee
            15: 13,  # left_ankle
            2: 14,   # right_eye
            1: 15,   # left_eye
            4: 16,   # right_ear
            3: 17,   # left_ear
        }

        for src_idx, dst_idx in coco_to_dw.items():
            if not _valid_coco(src_idx):
                continue
            x, y, z = coco_xyz[src_idx]
            u = (intrinsic_matrix[0, 0] * x / z) + intrinsic_matrix[0, 2]
            v = (intrinsic_matrix[1, 1] * y / z) + intrinsic_matrix[1, 2]
            out_candidate[dst_idx, 0] = u / float(width)
            out_candidate[dst_idx, 1] = v / float(height)
            out_score[dst_idx] = 1.0

        # Neck is derived from shoulders for OpenPose body18.
        has_l = _valid_coco(5)
        has_r = _valid_coco(6)
        neck = None
        if has_l and has_r:
            neck = 0.5 * (coco_xyz[5] + coco_xyz[6])
        elif has_l:
            neck = coco_xyz[5]
        elif has_r:
            neck = coco_xyz[6]
        if neck is not None and np.isfinite(neck).all() and neck[2] > 1e-6:
            u = (intrinsic_matrix[0, 0] * neck[0] / neck[2]) + intrinsic_matrix[0, 2]
            v = (intrinsic_matrix[1, 1] * neck[1] / neck[2]) + intrinsic_matrix[1, 2]
            out_candidate[1, 0] = u / float(width)
            out_candidate[1, 1] = v / float(height)
            out_score[1] = 1.0

        idx = np.arange(18, dtype=np.float32)
        visible = out_score > 0.3
        out_subset = np.where(visible, idx, -1.0).astype(np.float32)
        out_candidate[:, 0] = np.where(visible, out_candidate[:, 0], -1.0)
        out_candidate[:, 1] = np.where(visible, out_candidate[:, 1], -1.0)

        # Hands and face from mapped coco_wholebody layout.
        right_hand_3d = coco_xyz[112:133].astype(np.float32)
        left_hand_3d = coco_xyz[91:112].astype(np.float32)
        face_3d = coco_xyz[23:91].astype(np.float32)

        if coco_mask is not None and coco_mask.shape[0] >= 133:
            right_valid = coco_mask[112:133] > 0
            left_valid = coco_mask[91:112] > 0
            face_valid = coco_mask[23:91] > 0
            right_hand_3d[~right_valid] = 0.0
            left_hand_3d[~left_valid] = 0.0
            face_3d[~face_valid] = 0.0

        right_hand_xy, right_hand_score = self._project_points_3d(right_hand_3d, intrinsic_matrix, width, height)
        left_hand_xy, left_hand_score = self._project_points_3d(left_hand_3d, intrinsic_matrix, width, height)
        face_xy, face_score = self._project_points_3d(face_3d, intrinsic_matrix, width, height)

        # Fallback only when mapped points are mostly missing.
        left_shoulder = self._pick_joint(xyz_orig, [16, 17])
        right_shoulder = self._pick_joint(xyz_orig, [17, 16])
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_span = np.linalg.norm(left_shoulder - right_shoulder)
        else:
            neck_p = self._pick_joint(xyz_orig, [12])
            head_p = self._pick_joint(xyz_orig, [15])
            shoulder_span = np.linalg.norm(head_p - neck_p) * 2.0 if (neck_p is not None and head_p is not None) else 240.0

        if float(right_hand_score.sum()) < 6.0:
            rw = self._pick_joint(xyz_orig, [21, 20, 23])
            re = self._pick_joint(xyz_orig, [19, 18])
            synth = self._make_hand_points(rw, re, shoulder_span, is_right=True)
            if synth is not None:
                right_hand_xy, right_hand_score = self._project_points_3d(synth, intrinsic_matrix, width, height)
        if float(left_hand_score.sum()) < 6.0:
            lw = self._pick_joint(xyz_orig, [20, 21, 22])
            le = self._pick_joint(xyz_orig, [18, 19])
            synth = self._make_hand_points(lw, le, shoulder_span, is_right=False)
            if synth is not None:
                left_hand_xy, left_hand_score = self._project_points_3d(synth, intrinsic_matrix, width, height)
        if float(face_score.sum()) < 20.0:
            head_p = self._pick_joint(xyz_orig, [15])
            neck_p = self._pick_joint(xyz_orig, [12])
            synth = self._make_face_points(head_p, neck_p, shoulder_span)
            if synth is not None:
                face_xy, face_score = self._project_points_3d(synth, intrinsic_matrix, width, height)

        return out_candidate, out_subset, out_score, right_hand_xy, left_hand_xy, face_xy, right_hand_score, left_hand_score, face_score

    def _make_hand_points(self, wrist, elbow, shoulder_span, is_right):
        points = np.full((21, 3), 0.0, dtype=np.float32)
        if wrist is None or elbow is None:
            return None

        axis = wrist - elbow
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            axis = axis / axis_norm

        side = np.array([-axis[1], axis[0], 0.0], dtype=np.float32)
        side_norm = np.linalg.norm(side)
        if side_norm < 1e-6:
            side = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            side = side / side_norm

        # Use body scale so generated hand size tracks person size.
        scale = max(30.0, float(shoulder_span) * 0.11)
        lat_sign = -1.0 if is_right else 1.0

        finger_lateral = np.array([-0.30, -0.14, 0.0, 0.14, 0.30], dtype=np.float32) * lat_sign
        finger_base = np.array([0.18, 0.25, 0.30, 0.26, 0.20], dtype=np.float32)
        finger_step = np.array([0.18, 0.23, 0.26, 0.23, 0.20], dtype=np.float32)

        points[0] = wrist
        for f in range(5):
            for j in range(4):
                idx = 1 + f * 4 + j
                long = finger_base[f] + finger_step[f] * (j + 1)
                lat = finger_lateral[f] * (1.0 + 0.08 * (j + 1))
                points[idx] = wrist + scale * (long * axis + lat * side)
        return points

    def _make_face_points(self, head, neck, shoulder_span):
        points = np.full((68, 3), 0.0, dtype=np.float32)
        center = head if head is not None else neck
        if center is None:
            return None

        rx = max(24.0, float(shoulder_span) * 0.24)
        ry = max(30.0, float(shoulder_span) * 0.30)

        # Jawline (0-16)
        jaw_angles = np.linspace(np.deg2rad(200.0), np.deg2rad(-20.0), 17)
        points[0:17, 0] = center[0] + rx * np.cos(jaw_angles)
        points[0:17, 1] = center[1] + ry * np.sin(jaw_angles) + ry * 0.05
        points[0:17, 2] = center[2]

        # Eyebrows (17-26)
        brow_x = np.linspace(-0.42, -0.08, 5)
        points[17:22, 0] = center[0] + rx * brow_x
        points[17:22, 1] = center[1] - ry * (0.30 - 0.06 * np.sin(np.linspace(0, np.pi, 5)))
        points[17:22, 2] = center[2]

        brow_x_r = np.linspace(0.08, 0.42, 5)
        points[22:27, 0] = center[0] + rx * brow_x_r
        points[22:27, 1] = center[1] - ry * (0.30 - 0.06 * np.sin(np.linspace(0, np.pi, 5)))
        points[22:27, 2] = center[2]

        # Nose ridge/base (27-35)
        points[27:31, 0] = center[0]
        points[27:31, 1] = center[1] + np.linspace(-0.10 * ry, 0.20 * ry, 4)
        points[27:31, 2] = center[2]
        nose_angles = np.linspace(np.deg2rad(200.0), np.deg2rad(-20.0), 5)
        points[31:36, 0] = center[0] + 0.18 * rx * np.cos(nose_angles)
        points[31:36, 1] = center[1] + 0.24 * ry + 0.10 * ry * np.sin(nose_angles)
        points[31:36, 2] = center[2]

        # Eyes (36-47)
        eye_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        lcx, lcy = center[0] - 0.20 * rx, center[1] - 0.05 * ry
        rcx, rcy = center[0] + 0.20 * rx, center[1] - 0.05 * ry
        points[36:42, 0] = lcx + 0.12 * rx * np.cos(eye_angles)
        points[36:42, 1] = lcy + 0.07 * ry * np.sin(eye_angles)
        points[36:42, 2] = center[2]
        points[42:48, 0] = rcx + 0.12 * rx * np.cos(eye_angles)
        points[42:48, 1] = rcy + 0.07 * ry * np.sin(eye_angles)
        points[42:48, 2] = center[2]

        # Mouth (48-67)
        mouth_outer_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        mouth_inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        mcx, mcy = center[0], center[1] + 0.42 * ry
        points[48:60, 0] = mcx + 0.24 * rx * np.cos(mouth_outer_angles)
        points[48:60, 1] = mcy + 0.13 * ry * np.sin(mouth_outer_angles)
        points[48:60, 2] = center[2]
        points[60:68, 0] = mcx + 0.13 * rx * np.cos(mouth_inner_angles)
        points[60:68, 1] = mcy + 0.07 * ry * np.sin(mouth_inner_angles)
        points[60:68, 2] = center[2]

        return points

    def _infer_upstream_layout(self, xyz):
        n_src = int(xyz.shape[0])
        # Exact SMPL-X joint sets from JOINT_NAMES are typically 127 (no contour) or 144 (with contour).
        if n_src == 127 or n_src == 144:
            return "smplx_raw"

        # COCO-WholeBody style has 133 joints: body/feet(23) + face(68) + left hand(21) + right hand(21).
        # SMPL-X raw outputs are typically >=55 and often 127/144 joints.
        if n_src == 133:
            return "wholebody133"

        # Some SMPL-X exports can also be around 133 joints, so prefer SMPL-X when hand blocks 25:40/40:55
        # are spatially close to SMPL wrists 20/21.
        if n_src >= 55:
            lw = self._pick_joint(xyz, [20])
            rw = self._pick_joint(xyz, [21])
            ls = self._pick_joint(xyz, [16])
            rs = self._pick_joint(xyz, [17])
            if lw is not None and rw is not None and ls is not None and rs is not None and n_src >= 55:
                span = max(1.0, float(np.linalg.norm(ls - rs)))
                l_block = xyz[25:40]
                r_block = xyz[40:55]
                l_valid = np.isfinite(l_block).all(axis=1) & (l_block[:, 2] > 1e-6)
                r_valid = np.isfinite(r_block).all(axis=1) & (r_block[:, 2] > 1e-6)
                if int(l_valid.sum()) >= 8 and int(r_valid.sum()) >= 8:
                    l_dist = float(np.linalg.norm(l_block[l_valid] - lw[None, :], axis=1).mean())
                    r_dist = float(np.linalg.norm(r_block[r_valid] - rw[None, :], axis=1).mean())
                    if (l_dist / span) < 1.5 and (r_dist / span) < 1.5:
                        return "smplx_raw"

        if 133 <= n_src < 140:
            return "wholebody133"
        if n_src >= 55:
            return "smplx_raw"
        return "nlf24_or_small"

    def _extract_face_from_upstream(self, xyz):
        # Prefer direct 68-point face landmarks when available, with layout-aware index ranges.
        n_src = xyz.shape[0]
        layout = self._infer_upstream_layout(xyz)

        if layout == "wholebody133" and n_src >= 91:
            face_ranges = [(23, 91)]
        elif layout == "smplx_raw":
            # SMPL-X JOINT_NAMES layout:
            # 76:127 = face 51, and 127:144 = contour 17 (when enabled), so full face is 76:144.
            face_ranges = [(76, 144), (76, 127), (55, 123), (24, 92), (23, 91)]
        else:
            face_ranges = [(24, 92), (23, 91)]
        head = self._pick_joint(xyz, [15, 0])
        neck = self._pick_joint(xyz, [12])
        ref_center = head if head is not None else neck
        ref_scale = 200.0
        ls = self._pick_joint(xyz, [16])
        rs = self._pick_joint(xyz, [17])
        if ls is not None and rs is not None:
            ref_scale = max(50.0, float(np.linalg.norm(ls - rs)))

        best_face = None
        best_score = None
        for start, end in face_ranges:
            if n_src < end:
                continue
            face = xyz[start:end].astype(np.float32)
            valid = np.isfinite(face).all(axis=1) & (face[:, 2] > 1e-6)
            valid_count = int(valid.sum())
            if valid_count < 40:
                continue

            if ref_center is not None:
                center = face[valid].mean(axis=0)
                dist_norm = float(np.linalg.norm(center - ref_center) / ref_scale)
            else:
                dist_norm = 0.0

            score = dist_norm - (0.01 * valid_count)
            if best_score is None or score < best_score:
                best_score = score
                best_face = face

        return best_face

    def _extract_hand_from_upstream(self, xyz, is_right):
        n_src = xyz.shape[0]
        layout = self._infer_upstream_layout(xyz)

        # Prefer direct 21-point hand blocks only for COCO-WholeBody style ordering.
        if layout == "wholebody133" and n_src >= 133:
            start = 112 if is_right else 91
            hand = xyz[start:start + 21].astype(np.float32)
            valid = np.isfinite(hand).all(axis=1) & (hand[:, 2] > 1e-6)
            if int(valid.sum()) >= 10:
                return hand

        # For SMPL-X-like layouts, build 21 points from the native 15 hand joints.
        if layout == "smplx_raw" and n_src >= 55:
            wrist = self._pick_joint(xyz, [21, 20] if is_right else [20, 21])
            if wrist is None:
                return None

            # DW order: thumb, index, middle, ring, pinky.
            # Use true fingertip joints when available (SMPL-X JOINT_NAMES indices 66..75).
            if n_src >= 76:
                if is_right:
                    chains = [
                        (52, 53, 54, 71),  # thumb
                        (40, 41, 42, 72),  # index
                        (43, 44, 45, 73),  # middle
                        (49, 50, 51, 74),  # ring
                        (46, 47, 48, 75),  # pinky
                    ]
                else:
                    chains = [
                        (37, 38, 39, 66),  # thumb
                        (25, 26, 27, 67),  # index
                        (28, 29, 30, 68),  # middle
                        (34, 35, 36, 69),  # ring
                        (31, 32, 33, 70),  # pinky
                    ]
            else:
                base = 40 if is_right else 25
                if n_src < base + 15:
                    return None
                chains = [
                    (base + 12, base + 13, base + 14, None),  # thumb
                    (base + 0, base + 1, base + 2, None),     # index
                    (base + 3, base + 4, base + 5, None),     # middle
                    (base + 9, base + 10, base + 11, None),   # ring
                    (base + 6, base + 7, base + 8, None),     # pinky
                ]

            hand = np.full((21, 3), 0.0, dtype=np.float32)
            hand[0] = wrist
            out_i = 1
            for i0, i1, i2, i3 in chains:
                p1 = self._pick_joint(xyz, [i0])
                p2 = self._pick_joint(xyz, [i1])
                p3 = self._pick_joint(xyz, [i2])
                if p1 is None or p2 is None or p3 is None:
                    return None
                p4 = self._pick_joint(xyz, [i3]) if i3 is not None else None
                if p4 is None:
                    p4 = p3 + (p3 - p2)
                hand[out_i:out_i + 4] = np.stack([p1, p2, p3, p4], axis=0)
                out_i += 4
            return hand

        # Last-chance fallback for ambiguous layouts that still carry whole-body ordering.
        if n_src >= 133:
            start = 112 if is_right else 91
            hand = xyz[start:start + 21].astype(np.float32)
            valid = np.isfinite(hand).all(axis=1) & (hand[:, 2] > 1e-6)
            if int(valid.sum()) >= 10:
                return hand

        return None

    def _project_person_to_dw(self, person_xyz, intrinsic_matrix, width, height):
        # Match repo DW body order (OpenPose-18 style) instead of raw SMPL/NLF index order.
        # target indices: 0..17 = [nose/head, neck, l/r shoulder-elbow-wrist, l/r hip-knee-ankle, eyes/ears]
        out_candidate = np.full((18, 2), -1.0, dtype=np.float32)
        out_subset = np.full((18,), -1.0, dtype=np.float32)
        out_score = np.zeros((18,), dtype=np.float32)

        if person_xyz.shape[0] == 0:
            empty_hand = np.full((21, 2), -1.0, dtype=np.float32)
            empty_face = np.full((68, 2), -1.0, dtype=np.float32)
            return out_candidate, out_subset, out_score, empty_hand, empty_hand.copy(), empty_face, np.zeros((21,), dtype=np.float32), np.zeros((21,), dtype=np.float32), np.zeros((68,), dtype=np.float32)

        xyz = person_xyz.astype(np.float32)
        coco_xyz, coco_mask = self._to_coco_wholebody(xyz)
        if coco_xyz is not None:
            return self._project_person_to_dw_from_coco(xyz, coco_xyz, coco_mask, intrinsic_matrix, width, height)

        n_src = xyz.shape[0]

        # Source NLF/SMPL24 -> target DW body18.
        # 14..17 (eyes/ears) are unavailable from this mapping and stay invalid.
        src_to_dst = {
            15: 0,   # head -> nose/head slot
            12: 1,   # neck
            16: 2,   # left shoulder
            18: 3,   # left elbow
            20: 4,   # left wrist/hand
            17: 5,   # right shoulder
            19: 6,   # right elbow
            21: 7,   # right wrist/hand
            1: 8,    # left hip
            4: 9,    # left knee
            7: 10,   # left ankle/foot
            2: 11,   # right hip
            5: 12,   # right knee
            8: 13,   # right ankle/foot
        }

        for src_idx, dst_idx in src_to_dst.items():
            if src_idx >= n_src:
                continue
            x, y, z = xyz[src_idx]
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and z > 1e-6):
                continue

            u = (intrinsic_matrix[0, 0] * x / z) + intrinsic_matrix[0, 2]
            v = (intrinsic_matrix[1, 1] * y / z) + intrinsic_matrix[1, 2]
            out_candidate[dst_idx, 0] = u / float(width)
            out_candidate[dst_idx, 1] = v / float(height)
            out_score[dst_idx] = 1.0

        idx = np.arange(18, dtype=np.float32)
        visible = out_score > 0.3
        out_subset = np.where(visible, idx, -1.0).astype(np.float32)
        out_candidate[:, 0] = np.where(visible, out_candidate[:, 0], -1.0)
        out_candidate[:, 1] = np.where(visible, out_candidate[:, 1], -1.0)

        # Build hand/face points so RenderDWPose can draw them for SMPL/NLF-driven inputs.
        left_shoulder = self._pick_joint(xyz, [16, 17])
        right_shoulder = self._pick_joint(xyz, [17, 16])
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_span = np.linalg.norm(left_shoulder - right_shoulder)
        else:
            neck = self._pick_joint(xyz, [12])
            head = self._pick_joint(xyz, [15])
            shoulder_span = np.linalg.norm(head - neck) * 2.0 if (neck is not None and head is not None) else 240.0

        left_wrist = self._pick_joint(xyz, [20, 21, 22])
        right_wrist = self._pick_joint(xyz, [21, 20, 23])
        left_elbow = self._pick_joint(xyz, [18, 19])
        right_elbow = self._pick_joint(xyz, [19, 18])

        right_hand_3d = self._extract_hand_from_upstream(xyz, is_right=True)
        left_hand_3d = self._extract_hand_from_upstream(xyz, is_right=False)
        if right_hand_3d is None:
            right_hand_3d = self._make_hand_points(right_wrist, right_elbow, shoulder_span, is_right=True)
        if left_hand_3d is None:
            left_hand_3d = self._make_hand_points(left_wrist, left_elbow, shoulder_span, is_right=False)

        face_3d = self._extract_face_from_upstream(xyz)
        head = self._pick_joint(xyz, [15])
        neck = self._pick_joint(xyz, [12])
        if face_3d is None:
            face_3d = self._make_face_points(head, neck, shoulder_span)

        if right_hand_3d is not None:
            right_hand_xy, right_hand_score = self._project_points_3d(right_hand_3d, intrinsic_matrix, width, height)
        else:
            right_hand_xy = np.full((21, 2), -1.0, dtype=np.float32)
            right_hand_score = np.zeros((21,), dtype=np.float32)

        if left_hand_3d is not None:
            left_hand_xy, left_hand_score = self._project_points_3d(left_hand_3d, intrinsic_matrix, width, height)
        else:
            left_hand_xy = np.full((21, 2), -1.0, dtype=np.float32)
            left_hand_score = np.zeros((21,), dtype=np.float32)

        if face_3d is not None:
            face_xy, face_score = self._project_points_3d(face_3d, intrinsic_matrix, width, height)
        else:
            face_xy = np.full((68, 2), -1.0, dtype=np.float32)
            face_score = np.zeros((68,), dtype=np.float32)

        return out_candidate, out_subset, out_score, right_hand_xy, left_hand_xy, face_xy, right_hand_score, left_hand_score, face_score

    def convert(
        self,
        nlf_poses_world,
        width,
        height,
        camera_info_json="",
        camera_info_source="auto",
        cam_position_x=0.0,
        cam_position_y=-800.0,
        cam_position_z=-4000.0,
        cam_rotation_x_deg=0.0,
        cam_rotation_y_deg=0.0,
        cam_rotation_z_deg=0.0,
        camera_axis_correction="Zup->Yup",
        fov_degrees=55.0,
    ):
        # Share camera logic with ConvertWorldNLFPoseToCameraSpace.
        camera_converter = ConvertWorldNLFPoseToCameraSpace()
        nlf_poses_camera = camera_converter.convert(
            nlf_poses_world,
            camera_info_json=camera_info_json,
            camera_info_source=camera_info_source,
            cam_position_x=cam_position_x,
            cam_position_y=cam_position_y,
            cam_position_z=cam_position_z,
            cam_rotation_x_deg=cam_rotation_x_deg,
            cam_rotation_y_deg=cam_rotation_y_deg,
            cam_rotation_z_deg=cam_rotation_z_deg,
            camera_axis_correction=camera_axis_correction,
        )[0]

        pose_input = nlf_poses_camera["joints3d_nonparam"][0] if isinstance(nlf_poses_camera, dict) else nlf_poses_camera
        if not isinstance(pose_input, (list, tuple)):
            raise ValueError("Expected nlf_poses_world to contain a list of frame tensors.")

        render_fov_degrees = self._resolve_render_fov(fov_degrees, camera_info_json)
        intrinsic_matrix = self._build_preview_fov_intrinsic(width, height, render_fov_degrees)

        dwposes = []
        for frame in pose_input:
            if torch.is_tensor(frame):
                frame_t = frame.detach().float().cpu()
            else:
                frame_t = torch.tensor(frame, dtype=torch.float32)

            if frame_t.ndim == 2 and frame_t.shape[-1] == 3:
                frame_t = frame_t.unsqueeze(0)
            if frame_t.ndim != 3 or frame_t.shape[-1] != 3:
                raise ValueError(f"Expected frame shape [P,J,3] or [J,3], got {tuple(frame_t.shape)}")

            frame_np = frame_t.numpy()
            num_people = frame_np.shape[0]

            if num_people > 0:
                candidates = []
                subsets = []
                body_scores = []
                hands_list = []
                hand_scores_list = []
                faces_list = []
                face_scores_list = []
                for person_idx in range(num_people):
                    c, s, bs, rh, lh, fc, rhs, lhs, fcs = self._project_person_to_dw(frame_np[person_idx], intrinsic_matrix, width, height)
                    candidates.append(c)
                    subsets.append(s)
                    body_scores.append(bs)
                    # Keep the same order as VitPose path: right hand first, then left hand.
                    hands_list.extend([rh, lh])
                    hand_scores_list.extend([rhs, lhs])
                    faces_list.append(fc)
                    face_scores_list.append(fcs)

                bodies_candidate = np.stack(candidates, axis=0).astype(np.float32)
                bodies_subset = np.stack(subsets, axis=0).astype(np.float32)
                body_score = np.stack(body_scores, axis=0).astype(np.float32)
                hands = np.stack(hands_list, axis=0).astype(np.float32)
                hand_score = np.stack(hand_scores_list, axis=0).astype(np.float32)
                faces = np.stack(faces_list, axis=0).astype(np.float32)
                face_score = np.stack(face_scores_list, axis=0).astype(np.float32)
            else:
                bodies_candidate = np.zeros((0, 18, 2), dtype=np.float32)
                bodies_subset = np.zeros((0, 18), dtype=np.float32)
                body_score = np.zeros((0, 18), dtype=np.float32)
                hands = np.zeros((0, 21, 2), dtype=np.float32)
                faces = np.zeros((0, 68, 2), dtype=np.float32)
                hand_score = np.zeros((0, 21), dtype=np.float32)
                face_score = np.zeros((0, 68), dtype=np.float32)

            dwposes.append(
                {
                    "bodies": {
                        "candidate": bodies_candidate,
                        "subset": bodies_subset,
                    },
                    "hands": hands,
                    "faces": faces,
                    "body_score": body_score,
                    "hand_score": hand_score,
                    "face_score": face_score,
                }
            )

        return ({"poses": dwposes, "swap_hands": False},)

class PreviewNLFPoseGLBWithCamera:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "Path to an existing GLB (for example from SaveNLFPosesAs3D output_path)"}),
                "camera_info_json": ("STRING", {
                    "default": "{\"camera_to_world\": [[1, 0, 0, 0], [0, 1, 0, -0.8], [0, 0, 1, -4.0], [0, 0, 0, 1]], \"camera_unit_scale\": 1000.0, \"fov_degrees\": 55.0, \"frame_width\": 1280.0, \"frame_height\": 720.0}",
                    "multiline": True,
                    "tooltip": "Camera info edited by the viewport widget. Supports camera_to_world/world_to_camera formats used by ConvertWorldNLFPoseToCameraSpace. Optional camera_unit_scale scales camera translation into NLF world units.",
                }),
                "frame_width": ("INT", {"default": 1280, "min": 1, "max": 100000, "step": 1, "tooltip": "Viewfinder width reference (for example 1280)"}),
                "frame_height": ("INT", {"default": 720, "min": 1, "max": 100000, "step": 1, "tooltip": "Viewfinder height reference (for example 720)"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("camera_info_json", "glb_path",)
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Preview an existing NLF GLB in-node and output camera_info_json captured from the interactive viewport. GLB generation is handled by SaveNLFPosesAs3D. Frontend interaction pattern adapted from ComfyUI_Rabbit-Camera-Perspective and ComfyUI-qwenmultiangle."

    def _normalize_camera_info_json(self, camera_info_json):
        text = str(camera_info_json).strip() if camera_info_json is not None else ""
        if text == "":
            text = "{\"camera_to_world\": [[1, 0, 0, 0], [0, 1, 0, -0.8], [0, 0, 1, -4.0], [0, 0, 0, 1]], \"camera_unit_scale\": 1000.0, \"fov_degrees\": 55.0, \"frame_width\": 1280.0, \"frame_height\": 720.0}"
        try:
            data = json.loads(text)
        except Exception as e:
            raise ValueError(f"Invalid camera_info_json: {e}")
        if not isinstance(data, dict):
            raise ValueError("camera_info_json must decode to an object/dict.")
        if "frame_height" not in data and "frame_length" in data:
            data["frame_height"] = data["frame_length"]
        if "frame_width" not in data:
            data["frame_width"] = 1280.0
        if "frame_height" not in data:
            data["frame_height"] = 720.0
        return json.dumps(data, ensure_ascii=True)

    def _resolve_view_item_from_glb_path(self, glb_path):
        if glb_path is None or str(glb_path).strip() == "":
            raise ValueError("glb_path is empty. Connect SaveNLFPosesAs3D output_path.")

        abs_path = os.path.abspath(os.path.expanduser(str(glb_path).strip()))
        if not os.path.isfile(abs_path):
            raise ValueError(f"glb_path does not exist: {abs_path}")
        if not abs_path.lower().endswith(".glb"):
            raise ValueError(f"glb_path must point to a .glb file: {abs_path}")

        roots = [("output", folder_paths.get_output_directory())]
        if hasattr(folder_paths, "get_temp_directory"):
            try:
                roots.append(("temp", folder_paths.get_temp_directory()))
            except Exception:
                pass
        if hasattr(folder_paths, "get_input_directory"):
            try:
                roots.append(("input", folder_paths.get_input_directory()))
            except Exception:
                pass

        for root_type, root_dir in roots:
            if not root_dir:
                continue
            root_abs = os.path.abspath(root_dir)
            try:
                common = os.path.commonpath([abs_path, root_abs])
            except Exception:
                continue
            if common != root_abs:
                continue

            rel = os.path.relpath(abs_path, root_abs)
            filename = os.path.basename(rel)
            subfolder = os.path.dirname(rel).replace("\\", "/")
            return {
                "filename": filename,
                "subfolder": subfolder,
                "type": root_type,
                "abs_path": abs_path,
            }

        raise ValueError(
            "glb_path must be under ComfyUI output/temp/input directory so the /view endpoint can access it."
        )

    def preview(self, glb_path, camera_info_json, frame_width, frame_height):
        camera_json_out = self._normalize_camera_info_json(camera_info_json)
        camera_data = json.loads(camera_json_out)
        camera_data["frame_width"] = float(frame_width)
        camera_data["frame_height"] = float(frame_height)
        camera_json_out = json.dumps(camera_data, ensure_ascii=True)
        view_item = self._resolve_view_item_from_glb_path(glb_path)

        ui_payload = {
            "nlf_glb_preview": [{
                "filename": view_item["filename"],
                "subfolder": view_item["subfolder"],
                "type": view_item["type"],
                "camera_info_json": camera_json_out,
                "frame_width": float(frame_width),
                "frame_height": float(frame_height),
            }]
        }
        return {"ui": ui_payload, "result": (camera_json_out, view_item["abs_path"])}


class PreviewWorldNLFPoseWithCamera:
    # Edge pairs in source joint index space (matches NLF 24-joint layout used by renderer).
    NLF_EDGE_PAIRS = (
        (12, 17),  # neck -> right shoulder
        (12, 16),  # neck -> left shoulder
        (17, 19),  # right shoulder -> right elbow
        (19, 21),  # right elbow -> right wrist
        (16, 18),  # left shoulder -> left elbow
        (18, 20),  # left elbow -> left wrist
        (12, 2),   # neck -> right hip
        (2, 5),    # right hip -> right knee
        (5, 8),    # right knee -> right ankle
        (12, 1),   # neck -> left hip
        (1, 4),    # left hip -> left knee
        (4, 7),    # left knee -> left ankle
        (12, 15),  # neck -> head
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_poses_world": ("NLFPRED", {"tooltip": "World-space joints (for example LoadSMPLXNPZAsNLFPred.pose_results_all)"}),
                "camera_info_json": ("STRING", {
                    "default": "{\"camera_to_world\": [[1, 0, 0, 0], [0, 1, 0, -0.8], [0, 0, 1, -4.0], [0, 0, 0, 1]], \"camera_unit_scale\": 1000.0, \"fov_degrees\": 55.0, \"frame_width\": 1280.0, \"frame_height\": 720.0}",
                    "multiline": True,
                    "tooltip": "Camera info edited by the viewport widget. Supports camera_to_world/world_to_camera formats used by ConvertWorldNLFPoseToCameraSpace. Optional camera_unit_scale scales camera translation into NLF world units.",
                }),
                "frame_width": ("INT", {"default": 1280, "min": 1, "max": 100000, "step": 1, "tooltip": "Viewfinder width reference (for example 1280)"}),
                "frame_height": ("INT", {"default": 720, "min": 1, "max": 100000, "step": 1, "tooltip": "Viewfinder height reference (for example 720)"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 300.0, "step": 0.1, "tooltip": "Playback FPS for world-joint preview frames"}),
                "points_scale_factor": ("FLOAT", {"default": 1.0, "min": 0.000001, "max": 1000000.0, "step": 0.001, "tooltip": "Scale factor applied to all input NLF joint coordinates before preview rendering"}),
                "joint_radius": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 500.0, "step": 0.001, "tooltip": "Sphere radius used to draw each joint"}),
                "cylinder_radius": ("FLOAT", {"default": 14.0, "min": 0.001, "max": 500.0, "step": 0.001, "tooltip": "Cylinder radius used to draw each skeleton edge"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("camera_info_json", "glb_path",)
    OUTPUT_NODE = True
    FUNCTION = "preview_world"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Preview world-space NLFPRED joints directly in-node as spheres (joints) and cylinders (edges), while capturing camera_info_json from the interactive viewport."

    def _normalize_camera_info_json(self, camera_info_json):
        text = str(camera_info_json).strip() if camera_info_json is not None else ""
        if text == "":
            text = "{\"camera_to_world\": [[1, 0, 0, 0], [0, 1, 0, -0.8], [0, 0, 1, -4.0], [0, 0, 0, 1]], \"camera_unit_scale\": 1000.0, \"fov_degrees\": 55.0, \"frame_width\": 1280.0, \"frame_height\": 720.0}"
        try:
            data = json.loads(text)
        except Exception as e:
            raise ValueError(f"Invalid camera_info_json: {e}")
        if not isinstance(data, dict):
            raise ValueError("camera_info_json must decode to an object/dict.")
        if "frame_height" not in data and "frame_length" in data:
            data["frame_height"] = data["frame_length"]
        if "frame_width" not in data:
            data["frame_width"] = 1280.0
        if "frame_height" not in data:
            data["frame_height"] = 720.0
        return json.dumps(data, ensure_ascii=True)

    def _to_preview_frames(self, nlf_poses_world, points_scale_factor=1.0):
        if isinstance(nlf_poses_world, dict):
            pose_source = nlf_poses_world["joints3d_nonparam"][0] if "joints3d_nonparam" in nlf_poses_world else nlf_poses_world
        else:
            pose_source = nlf_poses_world

        if not isinstance(pose_source, (list, tuple)):
            raise ValueError("Expected nlf_poses_world to contain a list of frame tensors.")

        scale_factor = float(points_scale_factor)
        if not np.isfinite(scale_factor) or scale_factor <= 0.0:
            raise ValueError("points_scale_factor must be a positive finite number.")

        frames_payload = []
        for frame in pose_source:
            if torch.is_tensor(frame):
                frame_t = frame.detach().clone().float().cpu()
            else:
                frame_t = torch.tensor(copy.deepcopy(frame), dtype=torch.float32)

            if frame_t.ndim == 2 and frame_t.shape[-1] == 3:
                frame_t = frame_t.unsqueeze(0)
            if frame_t.ndim != 3 or frame_t.shape[-1] != 3:
                raise ValueError(f"Expected frame shape [P,J,3] or [J,3], got {tuple(frame_t.shape)}")

            frame_np = frame_t.numpy()
            people_payload = []
            for person_idx in range(frame_np.shape[0]):
                joints = frame_np[person_idx]
                joints_scaled = joints * scale_factor if scale_factor != 1.0 else joints
                points = joints_scaled.tolist()

                edges = []
                for a, b in self.NLF_EDGE_PAIRS:
                    if a >= joints.shape[0] or b >= joints.shape[0]:
                        continue
                    pa = joints[a]
                    pb = joints[b]
                    if not np.isfinite(pa).all() or not np.isfinite(pb).all():
                        continue
                    if float(np.sum(np.abs(pa))) < 1e-6 or float(np.sum(np.abs(pb))) < 1e-6:
                        continue
                    edges.append([int(a), int(b)])

                people_payload.append(
                    {
                        "points": points,
                        "edges": edges,
                    }
                )

            frames_payload.append(people_payload)

        return frames_payload

    def preview_world(self, nlf_poses_world, camera_info_json, frame_width, frame_height, fps, points_scale_factor, joint_radius, cylinder_radius):
        camera_json_out = self._normalize_camera_info_json(camera_info_json)
        camera_data = json.loads(camera_json_out)
        camera_data["frame_width"] = float(frame_width)
        camera_data["frame_height"] = float(frame_height)
        camera_json_out = json.dumps(camera_data, ensure_ascii=True)

        frames_payload = self._to_preview_frames(nlf_poses_world, points_scale_factor=points_scale_factor)
        ui_payload = {
            "nlf_world_preview": [
                {
                    "frames": frames_payload,
                    "fps": float(fps),
                    "points_scale_factor": float(points_scale_factor),
                    "joint_radius": float(joint_radius),
                    "edge_radius": float(cylinder_radius),
                    "cylinder_radius": float(cylinder_radius),
                    "camera_info_json": camera_json_out,
                    "frame_width": float(frame_width),
                    "frame_height": float(frame_height),
                }
            ]
        }

        return {"ui": ui_payload, "result": (camera_json_out, "")}


class PathStringToLoad3DModelFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path_string": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Absolute/relative model path. Converts to Load3D-compatible model_file text.",
                }),
                "default_type": (["auto", "input", "output", "temp"], {
                    "default": "auto",
                    "tooltip": "Type annotation used when path_string is relative and unannotated.",
                }),
            },
            "optional": {
                "must_exist": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, raise an error when the resolved path does not exist.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_file", "absolute_path")
    FUNCTION = "convert"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Convert a path string to a Load3D/Preview3D model_file value (combo-like annotated path)."

    @staticmethod
    def _normalize_slashes(p: str) -> str:
        return p.replace("\\", "/")

    @staticmethod
    def _split_known_roots(abs_path: str):
        roots = [
            ("input", folder_paths.get_input_directory()),
            ("output", folder_paths.get_output_directory()),
        ]

        if hasattr(folder_paths, "get_temp_directory"):
            try:
                roots.append(("temp", folder_paths.get_temp_directory()))
            except Exception:
                pass

        for root_type, root_dir in roots:
            if not root_dir:
                continue
            root_abs = os.path.abspath(root_dir)
            try:
                common = os.path.commonpath([abs_path, root_abs])
            except Exception:
                continue
            if common != root_abs:
                continue

            rel = os.path.relpath(abs_path, root_abs)
            rel = PathStringToLoad3DModelFile._normalize_slashes(rel)
            return rel, root_type

        return None, None

    def convert(self, path_string, default_type="auto", must_exist=True):
        raw = str(path_string).strip()
        if raw == "":
            raise ValueError("path_string is empty.")

        name, annotated_base = folder_paths.annotated_filepath(raw)
        name = self._normalize_slashes(name.strip())

        if annotated_base is not None:
            root_type = "input"
            try:
                if annotated_base == folder_paths.get_output_directory():
                    root_type = "output"
                elif hasattr(folder_paths, "get_temp_directory") and annotated_base == folder_paths.get_temp_directory():
                    root_type = "temp"
            except Exception:
                pass

            abs_path = os.path.abspath(os.path.join(annotated_base, name))
            model_file = f"{name} [{root_type}]"
            if must_exist and not os.path.isfile(abs_path):
                raise ValueError(f"Resolved annotated path does not exist: {abs_path}")
            return (model_file, abs_path)

        if os.path.isabs(raw):
            abs_path = os.path.abspath(raw)
            rel, root_type = self._split_known_roots(abs_path)
            if rel is None or root_type is None:
                raise ValueError(
                    "Absolute path must be under ComfyUI input/output/temp to produce a combo-like model_file value."
                )
            model_file = f"{rel} [{root_type}]"
            if must_exist and not os.path.isfile(abs_path):
                raise ValueError(f"Path does not exist: {abs_path}")
            return (model_file, abs_path)

        # Relative unannotated input: apply default annotation.
        chosen_type = default_type if default_type in ("input", "output", "temp") else "input"
        base = folder_paths.get_input_directory()
        if chosen_type == "output":
            base = folder_paths.get_output_directory()
        elif chosen_type == "temp" and hasattr(folder_paths, "get_temp_directory"):
            try:
                base = folder_paths.get_temp_directory()
            except Exception:
                base = folder_paths.get_input_directory()

        model_file = f"{name} [{chosen_type}]"
        abs_path = os.path.abspath(os.path.join(base, name))
        if must_exist and not os.path.isfile(abs_path):
            raise ValueError(f"Resolved relative path does not exist: {abs_path}")
        return (model_file, abs_path)


NODE_CLASS_MAPPINGS = {
    "PoseDetectionVitPoseToDWPose": PoseDetectionVitPoseToDWPose,
    "RenderNLFPoses": RenderNLFPoses,
    "RenderNLFPosesWithCameraInfo": RenderNLFPosesWithCameraInfo,
    "RenderDWPose": RenderDWPose,
    "RenderDWPoseWithCameraInfo": RenderDWPoseWithCameraInfo,
    "ConvertOpenPoseKeypointsToDWPose": ConvertOpenPoseKeypointsToDWPose,
    "SaveNLFPosesAs3D": SaveNLFPosesAs3D,
    "LoadSMPLXNPZAsNLFPred": LoadSMPLXNPZAsNLFPred,
    "ConvertWorldNLFPoseToCameraSpace": ConvertWorldNLFPoseToCameraSpace,
    "ConvertWorldNLFPoseToDWPose": ConvertWorldNLFPoseToDWPose,
    "PreviewNLFPoseGLBWithCamera": PreviewNLFPoseGLBWithCamera,
    "PreviewWorldNLFPoseWithCamera": PreviewWorldNLFPoseWithCamera,
    "PathStringToLoad3DModelFile": PathStringToLoad3DModelFile,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseDetectionVitPoseToDWPose": "Pose Detection VitPose to DWPose",
    "RenderNLFPoses": "Render NLF Poses",
    "RenderNLFPosesWithCameraInfo": "Render NLF Poses with Camera Info",
    "RenderDWPose": "Render DW Poses",
    "RenderDWPoseWithCameraInfo": "Render DW Poses with Camera Info",
    "ConvertOpenPoseKeypointsToDWPose": "Convert OpenPose Keypoints to DWPose",
    "SaveNLFPosesAs3D": "Save NLF Poses as 3D Animation",
    "LoadSMPLXNPZAsNLFPred": "Load SMPL-X NPZ as NLF Poses",
    "ConvertWorldNLFPoseToCameraSpace": "Convert World NLF Poses to Camera Space",
    "ConvertWorldNLFPoseToDWPose": "Convert World NLF Poses to DWPose",
    "PreviewNLFPoseGLBWithCamera": "Preview NLF Pose GLB with Camera",
    "PreviewWorldNLFPoseWithCamera": "Preview World NLF Pose with Camera",
    "PathStringToLoad3DModelFile": "Path String to Load3D Model File",
}
