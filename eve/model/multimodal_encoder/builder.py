import os
from .clip_encoder import CLIPVisionTower
from .siglip_vit import create_siglip_vit
from .whisper_tower_hg import WhisperAudioTower
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 37:
    from .siglip_encoder import SiglipVisionTower
    
def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    is_absolute_path_exists = os.path.exists(image_tower)
    if is_absolute_path_exists:
        vision_tower_type = getattr(image_tower_cfg, 'vision_tower_type', None)
        if not vision_tower_type:
            return CLIPVisionTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
        if vision_tower_type == "clip":
            return CLIPVisionTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
        elif vision_tower_type == "siglip":
            vision_tower_params = dict(
            model_name="siglip_large_patch16_384",
            select_feature="same",
            image_size=384,
            pixel_mean=(0.5, 0.5, 0.5),
            pixel_std=(0.5, 0.5, 0.5),
            select_layer=-1,
            ckpt_path=image_tower,
            )
            
            return create_siglip_vit(**vision_tower_params)
        
        return CLIPVisionTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    if image_tower.startswith("openai") or image_tower.startswith("laion"):
        return CLIPVisionTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    if image_tower.startswith("google"):
        return SiglipVisionTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    # if image_tower.endswith('LanguageBind_Image'):
    #     return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')

# def build_video_tower(video_tower_cfg, **kwargs):
#     video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
#     if video_tower.endswith('LanguageBind_Video_merge'):
#         return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
#     raise ValueError(f'Unknown video tower: {video_tower}')
# ============================================================================================================


def build_audio_tower(audio_tower_cfg, **kwargs):
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))
    is_absolute_path_exists = os.path.exists(audio_tower)
    if is_absolute_path_exists:
        audio_tower_type = getattr(audio_tower_cfg, 'audio_tower_type', 'whisper')
        if audio_tower_type == "whisper":
            return WhisperAudioTower(audio_tower, args=audio_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown audio tower: {audio_tower_type}')