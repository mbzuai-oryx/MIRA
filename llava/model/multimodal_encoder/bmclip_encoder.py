import open_clip
import torch
import torch.nn as nn
from typing import Optional, List
from llava.utils import rank0_print
import json
import torchvision
from transformers import CLIPImageProcessor
from open_clip.transformer import _expand_token
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

class SimpleImageProcessor:
    """包装 open_clip 的 preprocess 函数，使其接口类似 CLIPImageProcessor"""
    def __init__(self, preprocess):
        self.prepc = preprocess

    def preprocess(self, images, return_tensors="pt", **kwargs):
        if return_tensors != "pt":
            raise NotImplementedError

        toret = None
        if isinstance(images, list):
            toret = [self.prepc(img).unsqueeze(0) for img in images]
        else:
            toret = self.prepc(images).unsqueeze(0)
        
        return {"pixel_values": toret}

class BMClipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.model_name = vision_tower.replace("open_clip_hub:", "")
        self.pretrained = args.vision_tower_pretrained

        # Load model based on conditions, similar to CLIPVisionTower
        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        # No config preloading; properties use defaults if not loaded

    def load_model(self, device_map=None):
        """Load the BiomedClip model using open_clip."""
        if self.is_loaded:
            rank0_print(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return

        with open("/home/jinhong.wang/workdir/checkpoints/released/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json", "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]

        model_name = "biomedclip_local"
        if (not model_name.startswith(HF_HUB_PREFIX)
            and model_name not in _MODEL_CONFIGS
            and config is not None):
            _MODEL_CONFIGS[model_name] = model_cfg

        # model, _, preprocess = create_model_and_transforms(
        #     model_name=model_name,
        #     pretrained="checkpoints/open_clip_pytorch_model.bin",
        #     **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        # )

        model, _, preprocess = open_clip.create_model_and_transforms(model_name="biomedclip_local", pretrained="/home/jinhong.wang/workdir/checkpoints/released/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin", precision="bf16", device="cuda", **{f"image_{k}": v for k, v in preprocess_cfg.items()})

        # Load model and preprocessor from open_clip
        # model, _, preprocess = open_clip.create_model_and_transforms(
        #     'ViT-B-16', pretrained=self.vision_tower_name
        # )
        # model, _, preprocess = open_clip.create_model_and_transforms(model_name=self.model_name, pretrained=self.pretrained, precision="fp32", device="cuda")
        self.vision_tower = model.visual  # Vision component of BiomedClip
        self.image_processor = SimpleImageProcessor(preprocess)  # For external preprocessing
        self.preprocess = self.image_processor

        self.device_setting = device_map
        self.precision_setting = torch.bfloat16

        if device_map is not None:
            self.vision_tower.to(device_map)
        self.vision_tower.requires_grad_(False)  # Freeze weights
        self.is_loaded = True

    def get_hidden_states(self, images):
        hidden_states = []

        def hook_fn(module, input, output):
            hidden_states.append(output)

        # Register hooks on each block in the VisionTransformer's blocks
        hooks = []
        for block in self.vision_tower.trunk.blocks:
            hook = block.register_forward_hook(hook_fn)
            hooks.append(hook)

        # Run the forward pass through TimmModel
        with torch.no_grad():
            _ = self.vision_tower(images)  # Triggers hooks

        # Remove hooks after use
        for hook in hooks:
            hook.remove()

        # The final output after the norm layer (before head) is available via forward_features
        with torch.no_grad():
            final_output = self.vision_tower.trunk.norm(hidden_states[-1])
            hidden_states.append(final_output)

        return hidden_states

    def feature_select(self, hidden_states):
        """Select features from hidden states based on configuration."""
        select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(hidden_states) // 4
            image_features = torch.cat(
                [hidden_states[i] for i in range(select_every_k_layer + self.select_layer, len(hidden_states), select_every_k_layer)],
                dim=-1
            )
            select_feature_type = select_feature_type.replace("slicefour_", "")
        elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
            select_layers = [-2, -5, -8, -11, 6]  # Negative indices count from the end
            image_features = torch.cat(
                [hidden_states[i] for i in select_layers],
                dim=-1
            )
            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
        else:
            image_features = hidden_states[self.select_layer]

        if select_feature_type == "patch":
            image_features = image_features[:, 1:]  # Exclude CLS token
        elif select_feature_type == "cls_patch":
            image_features = image_features  # Include CLS token
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features

    def forward(self, images):
        """Process images or a list of images to extract selected features."""
        if isinstance(images, list):
            image_features = []
            for image in images:
                hidden_states = self.get_hidden_states(image.unsqueeze(0).to(self.device))
                image_feature = self.feature_select(hidden_states).to(image.dtype)
                image_features.append(image_feature)
            return image_features
        else:
            hidden_states = self.get_hidden_states(images.to(self.device))
            return self.feature_select(hidden_states).to(images.dtype)

    @property
    def dummy_feature(self):
        """Return a zero tensor for initialization purposes."""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """Return the data type of the model parameters."""
        return self.precision_setting
        # return next(self.vision_tower.parameters()).dtype if self.is_loaded else torch.float32

    @property
    def device(self):
        """Return the device of the model parameters."""
        return self.device_setting
        # return next(self.vision_tower.parameters()).device if self.is_loaded else torch.device("cpu")

    @property
    def hidden_size(self):
        """Return the embedding dimension, adjusted for multi-layer selection."""
        # _hidden_size = self.vision_tower.embed_dim if self.is_loaded else 768  # ViT-B/16 default
        _hidden_size = 768
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        if "slice_m25811_f6" in self.select_feature:
            _hidden_size *= 5
        return _hidden_size

    @property
    def num_patches_per_side(self):
        """Return patches per side (224 / 16)."""
        return 14  # Hardcoded for ViT-B/16 with image_size=224, patch_size=16

    @property
    def num_patches(self):
        """Return total number of patches, including CLS token if applicable."""
        _num_patches = 196  # 14 * 14
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def image_size(self):
        """Return the input image size."""
        return 224  # Hardcoded for ViT-B/16

    @property
    def config(self):
        return SimpleConfig()

class SimpleConfig:
    def __init__(self, hs=768, ims=224, ps=16):
        self.hidden_size = hs
        self.image_size = ims
        self.patch_size = ps

# class BMClipProcessor(nn.Module):
#     def __init__(self, main_func):
#         super().__init__()
#         self.process_func = main_func

#     def preprocess(self, image, return_tensors='pt'):
#         if return_tensors != 'pt':
#             raise NotImplementedError

#         return_dict = {"pixel_values":[]}
#         return_dict["pixel_values"].append(torch.tensor(self.process_func(image)))

#         return return_dict
        

# class BMClipVisionTower(nn.Module):
#     def __init__(self, vision_tower, args, delay_load=False):
#         super().__init__()

#         self.is_loaded = False
#         self.model_name = vision_tower.replace("open_clip_hub:", "")
#         self.pretrained = args.vision_tower_pretrained
#         self.select_layer = args.mm_vision_select_layer
#         self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

#         if not delay_load:
#             rank0_print(f"Loading vision tower: {vision_tower}")
#             self.load_model()
#         elif getattr(args, "unfreeze_mm_vision_tower", False):
#             # TODO: better detector is needed.
#             rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
#             self.load_model()
#         elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
#             rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
#             self.load_model()

#     def load_model(self, device_map="auto"):
#         rank0_print(f"Loading OpenCLIP model: {self.model_name}")
#         rank0_print(f"Pretrained: {self.pretrained}")
#         # vision_tower, _, image_processor = open_clip.create_model_and_transforms(model_name=self.model_name, pretrained=self.pretrained, precision="fp32", device="cuda")

#         with open("/home/jinhong.wang/workdir/checkpoints/released/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json", "r") as f:
#             config = json.load(f)
#             model_cfg = config["model_cfg"]
#             preprocess_cfg = config["preprocess_cfg"]

#         model_name = "biomedclip_local"
#         if (not model_name.startswith(HF_HUB_PREFIX)
#             and model_name not in _MODEL_CONFIGS
#             and config is not None):
#             _MODEL_CONFIGS[model_name] = model_cfg

#         tokenizer = get_tokenizer(model_name)

#         # model, _, preprocess = create_model_and_transforms(
#         #     model_name=model_name,
#         #     pretrained="checkpoints/open_clip_pytorch_model.bin",
#         #     **{f"image_{k}": v for k, v in preprocess_cfg.items()},
#         # )

#         vision_tower, _, image_processor = open_clip.create_model_and_transforms(model_name="biomedclip_local", pretrained="/home/jinhong.wang/workdir/checkpoints/released/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin", precision="fp32", device="cuda", **{f"image_{k}": v for k, v in preprocess_cfg.items()})

#         resize_transform = [t for t in image_processor.transforms if isinstance(t, torchvision.transforms.Resize)][0]
#         normalize_transform = [t for t in image_processor.transforms if isinstance(t, torchvision.transforms.Normalize)][0]
#         self.resize_transform_size = resize_transform.size  # 224 or 384
#         # self.patch_size = vision_tower.visual.conv1.kernel_size[0]
#         self.patch_size=16  # 14 or 16

#         self.image_processor = CLIPImageProcessor.from_pretrained(
#             "openai/clip-vit-large-patch14",
#             crop_size=resize_transform.size,
#             size={"shortest_edge": resize_transform.size},
#             image_mean=list(normalize_transform.mean),
#             image_std=list(normalize_transform.std),
#         )
#         rank0_print(f"Loaded image processor: {self.image_processor}")
#         self.vision_tower = vision_tower.visual
#         self.vision_tower.requires_grad_(False)

#         self.is_loaded = True

#     def feature_select(self, hidden_states: List[torch.Tensor]):
#         select_feature_type = self.select_feature
        
#         # Handle different layer selection strategies
#         if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
#             select_every_k_layer = len(hidden_states) // 4
#             selected_layers = [
#                 hidden_states[i] 
#                 for i in range(select_every_k_layer + self.select_layer, 
#                               len(hidden_states), 
#                               select_every_k_layer)
#             ]
#             image_features = torch.cat(selected_layers, dim=-1)
#             select_feature_type = select_feature_type.replace("slicefour_", "")
#         elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
#             select_layers = [-2, -5, -8, -11, 6]
#             image_features = torch.cat([hidden_states[i] for i in select_layers], dim=-1)
#             select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
#         else:
#             image_features = hidden_states[self.select_layer]
        
#         # Handle feature type selection
#         if select_feature_type == "patch":
#             image_features = image_features[:, 1:]  # Remove CLS token
#         elif select_feature_type == "cls_patch":
#             image_features = image_features  # Keep CLS token
#         else:
#             raise ValueError(f"Invalid feature type: {select_feature_type}")
        
#         return image_features

#     def forward(self, images: torch.Tensor):
#         # Reset cache and process inputs
#         self.hidden_states_cache = []
#         self.initial_embedding = None
        
#         if isinstance(images, list):
#             image_features = []
#             for image in images:
#                 self.hidden_states_cache = []
#                 self.initial_embedding = None
#                 img_tensor = image.to(self.device, dtype=self.dtype).unsqueeze(0)
#                 _ = self.vision_tower(img_tensor)
                
#                 # Build full hidden states list
#                 full_hidden = []
#                 if self.initial_embedding is not None:
#                     full_hidden.append(self.initial_embedding)
#                 full_hidden.extend(self.hidden_states_cache)
                
#                 # Select features
#                 img_feat = self.feature_select(full_hidden).to(image.dtype)
#                 image_features.append(img_feat)
#             return image_features
#         else:
#             img_tensor = images.to(self.device, dtype=self.dtype)
#             _ = self.vision_tower(img_tensor)
            
#             # Build full hidden states list
#             full_hidden = []
#             if self.initial_embedding is not None:
#                 full_hidden.append(self.initial_embedding)
#             full_hidden.extend(self.hidden_states_cache)
            
#             return self.feature_select(full_hidden).to(images.dtype)

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return self.vision_tower.conv1.weight.dtype

#     @property
#     def device(self):
#         return self.vision_tower.conv1.weight.device

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.vision_tower.config
#         else:
#             return self.cfg_only

#     @property
#     def hidden_size(self):
#         base_size = 768
#         if "slicefour" in self.select_feature:
#             return base_size * 4
#         if "slice_m25811_f6" in self.select_feature:
#             return base_size * 5
#         return base_size

#     @property
#     def num_patches_per_side(self):
#         return self.config.image_size // self.config.patch_size

#     @property
#     def num_patches(self):
#         return (self.config.image_size // self.config.patch_size) ** 2 + \
#                (1 if "cls_patch" in self.select_feature else 0)

#     @property
#     def image_size(self):
#         return self.config.image_size


# import torch
# import torch.nn as nn
# from llava.utils import rank0_print
# from open_clip import create_model_from_pretrained, get_tokenizer
# # from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig



# class CLIPVisionTower(nn.Module):
#     def __init__(self, vision_tower, args, delay_load=False):
#         super().__init__()

#         self.is_loaded = False

#         self.vision_tower_name = vision_tower
#         self.select_layer = args.mm_vision_select_layer
#         self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

#         if not delay_load:
#             rank0_print(f"Loading vision tower: {vision_tower}")
#             self.load_model()
#         elif getattr(args, "unfreeze_mm_vision_tower", False):
#             # TODO: better detector is needed.
#             rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
#             self.load_model()
#         elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
#             rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
#             self.load_model()
#         else:
#             self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

#     def load_model(self, device_map=None):
#         if self.is_loaded:
#             rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
#             return

#         # self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#         # self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
#         self.vision_tower, self.image_processor = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#         # tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#         self.vision_tower.requires_grad_(False)
#         self.is_loaded = True

#     def feature_select(self, image_forward_outs):
#         select_feature_type = self.select_feature

#         if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
#             select_every_k_layer = len(image_forward_outs.hidden_states) // 4
#             image_features = torch.cat([image_forward_outs.hidden_states[i] for i in range(select_every_k_layer + self.select_layer, len(image_forward_outs.hidden_states), select_every_k_layer)], dim=-1)
#             select_feature_type = select_feature_type.replace("slicefour_", "")
#         elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
#             select_layers = [-2, -5, -8, -11, 6]
#             image_features = torch.cat([image_forward_outs.hidden_states[i] for i in select_layers], dim=-1)
#             select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
#         else:
#             image_features = image_forward_outs.hidden_states[self.select_layer]

#         if select_feature_type == "patch":
#             image_features = image_features[:, 1:]
#         elif select_feature_type == "cls_patch":
#             image_features = image_features
#         else:
#             raise ValueError(f"Unexpected select feature: {select_feature_type}")
#         return image_features

#     def forward(self, images):
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
#                 image_feature = self.feature_select(image_forward_out).to(image.dtype)
#                 image_features.append(image_feature)
#         else:
#             image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
#             image_features = self.feature_select(image_forward_outs).to(images.dtype)

#         return image_features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return self.vision_tower.dtype

#     @property
#     def device(self):
#         return self.vision_tower.device

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.vision_tower.config
#         else:
#             return self.cfg_only

#     @property
#     def hidden_size(self):
#         _hidden_size = self.config.hidden_size
#         if "slicefour" in self.select_feature:
#             _hidden_size *= 4
#         if "slice_m25811_f6" in self.select_feature:
#             _hidden_size *= 5
#         return _hidden_size

#     @property
#     def num_patches_per_side(self):
#         return self.config.image_size // self.config.patch_size

#     @property
#     def num_patches(self):
#         _num_patches = (self.config.image_size // self.config.patch_size) ** 2
#         if "cls_patch" in self.select_feature:
#             _num_patches += 1
#         return _num_patches

#     @property
#     def image_size(self):
#         return self.config.image_size