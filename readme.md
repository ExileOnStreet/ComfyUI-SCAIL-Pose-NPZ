# ComfyUI nodes for SCAIL-pose processing


The code is cleaned, simplified version of: https://github.com/zai-org/SCAIL-Pose

For face and hands, instead of DWPose this uses Vitpose and it's outputs converted into DWpose format for the optional alignment

VitPose detector is available in these nodes: https://github.com/kijai/ComfyUI-WanAnimatePreprocess

NLF model (to ComfyUI/models/nlf)

https://huggingface.co/Kijai/WanVideo_comfy/blob/main/SCAIL/nlf_l_multi_0.3.2_fp16.safetensors

