"""
Node registry - collects all nodes from this package
"""

from .lora_loader_by_index import LoraLoaderByIndex
from .lora_loader_filtered import LoraLoaderFiltered
from .image_loader_by_index import ImageLoaderByIndex
from .video_loader_by_index import VideoLoaderByIndex
from .video_scrubber import VideoScrubber
from .conditioning_normalizer import ConditioningNormalizer
from .conditioning_scale import ConditioningScale
from .conditioning_clamp import ConditioningClamp
from .conditioning_stats import ConditioningStats
from .conditioning_utils import ConditioningLerp, ConditioningSubtract, RandomSelect
from .lora_list_directory import LoraListDirectory
from .string_concatenate import StringConcatenate
from .string_merge_unique import StringMergeUnique
from .string_replacer import StringReplacer
from .string_extractor import StringExtractor
from .string_list_slicer import StringListSlicer
from .string_splitter import StringSplitter
from .print_node import Print
from .latent_utils import LatentClamp, LatentScale, LatentNormalize, LatentStats
from .latent_channel import LatentChannelOffset, LatentChannelOffset16, LatentChannelScale, LatentChannelScale16
from .prompt_builder import PromptBuilder
from .prompt_builder_b import PromptBuilderB
from .prompt_store import PromptStore, PromptStoreB, PromptStoreClear, PromptStoreCustom, PromptStoreGet, PromptStoreHeadings, PromptStoreList
from .emphasis_encode import EmphasisEncode, EmphasisEncodeAdvanced
from .inspect_tensor import InspectTensor
from .image_utils import ResizeByShorterEdge, ResizeToMatch, ImageBlend
from .dimension_scale import DimensionScale
from .json_utils import LoadJSONFile, LoadJSONFilePath
from .token_counter import TokenCounter, ConditioningTokenCount
from .token_remap import TokenRemap, ClipRemap, TokenInspector
from .concept_remap import ConceptRemap
from .hyper_remap import HyperRemap
from .hyper_remap_slim import HyperRemapSlim
from .conditioning_projection_removal import ConditioningProjectionRemoval
from .neutral_prompt import NeutralPrompt
from .neutral_prompt_guider import NeutralPromptEntry, NeutralPromptEmpty, NeutralPromptGuider
from .compound_prompt import CompoundPrompt
from .lora_scale_save import LoraScaleSave
from .lora_truncate_rank import LoraTruncateRank
from .lora_metadata import LoraMetadata
from .safetensors_metadata import SafetensorsMetadata
from .float_lerp import FloatLerp
from .curve_sample import CurveSample
from .curve_definition import CurveDefinition
from .curve_cfg_guider import CurveCFGGuider
from .curve_from_core import CurveFromCore
from .curve_to_core import CurveToCore
from .cfg_zero_star_scaled import CFGZeroStarScaled
from .cropped_image import CroppedImage
from .image_inset import ImageInset
from .hyper_remap_krea2edit import HyperRemapKrea2Edit
from .hyper_remap_krea2edit_slim import HyperRemapKrea2EditSlim

NODE_CLASS_MAPPINGS = {
    "CCN_LoraLoaderByIndex": LoraLoaderByIndex,
    "CCN_LoraLoaderFiltered": LoraLoaderFiltered,
    "CCN_ImageLoaderByIndex": ImageLoaderByIndex,
    "CCN_VideoLoaderByIndex": VideoLoaderByIndex,
    "CCN_VideoScrubber": VideoScrubber,
    "CCN_ConditioningNormalizer": ConditioningNormalizer,
    "CCN_ConditioningScale": ConditioningScale,
    "CCN_ConditioningClamp": ConditioningClamp,
    "CCN_ConditioningStats": ConditioningStats,
    "CCN_ConditioningLerp": ConditioningLerp,
    "CCN_ConditioningSubtract": ConditioningSubtract,
    "CCN_LoraListDirectory": LoraListDirectory,
    "CCN_StringConcatenate": StringConcatenate,
    "CCN_StringMergeUnique": StringMergeUnique,
    "CCN_StringReplacer": StringReplacer,
    "CCN_StringExtractor": StringExtractor,
    "CCN_StringListSlicer": StringListSlicer,
    "CCN_StringSplitter": StringSplitter,
    "CCN_Print": Print,
    "CCN_LatentClamp": LatentClamp,
    "CCN_LatentScale": LatentScale,
    "CCN_LatentNormalize": LatentNormalize,
    "CCN_LatentStats": LatentStats,
    "CCN_LatentChannelOffset": LatentChannelOffset,
    "CCN_LatentChannelOffset16": LatentChannelOffset16,
    "CCN_LatentChannelScale": LatentChannelScale,
    "CCN_LatentChannelScale16": LatentChannelScale16,
    "CCN_PromptBuilder": PromptBuilder,
    "CCN_PromptBuilderB": PromptBuilderB,
    "CCN_PromptStore": PromptStore,
    "CCN_PromptStoreB": PromptStoreB,
    "CCN_PromptStoreClear": PromptStoreClear,
    "CCN_PromptStoreCustom": PromptStoreCustom,
    "CCN_PromptStoreGet": PromptStoreGet,
    "CCN_PromptStoreHeadings": PromptStoreHeadings,
    "CCN_PromptStoreList": PromptStoreList,
    "CCN_EmphasisEncode": EmphasisEncode,
    "CCN_EmphasisEncodeAdvanced": EmphasisEncodeAdvanced,
    "CCN_InspectTensor": InspectTensor,
    "CCN_RandomSelect": RandomSelect,
    "CCN_ResizeByShorterEdge": ResizeByShorterEdge,
    "CCN_ResizeToMatch": ResizeToMatch,
    "CCN_ImageBlend": ImageBlend,
    "CCN_LoadJSONFile": LoadJSONFile,
    "CCN_LoadJSONFilePath": LoadJSONFilePath,
    "CCN_TokenCounter": TokenCounter,
    "CCN_ConditioningTokenCount": ConditioningTokenCount,
    "CCN_TokenRemap": TokenRemap,
    "CCN_ClipRemap": ClipRemap,
    "CCN_TokenInspector": TokenInspector,
    "CCN_ConceptRemap": ConceptRemap,
    "CCN_HyperRemap": HyperRemap,
    "CCN_HyperRemapSlim": HyperRemapSlim,
    "CCN_ConditioningProjectionRemoval": ConditioningProjectionRemoval,
    "CCN_NeutralPrompt": NeutralPrompt,
    "CCN_NeutralPromptEntry": NeutralPromptEntry,
    "CCN_NeutralPromptEmpty": NeutralPromptEmpty,
    "CCN_NeutralPromptGuider": NeutralPromptGuider,
    "CCN_CompoundPrompt": CompoundPrompt,
    "CCN_LoraScaleSave": LoraScaleSave,
    "CCN_LoraTruncateRank": LoraTruncateRank,
    "CCN_LoraMetadata": LoraMetadata,
    "CCN_SafetensorsMetadata": SafetensorsMetadata,
    "CCN_DimensionScale": DimensionScale,
    "CCN_FloatLerp": FloatLerp,
    "CCN_CurveSample": CurveSample,
    "CCN_CurveDefinition": CurveDefinition,
    "CCN_CurveCFGGuider": CurveCFGGuider,
    "CCN_CurveFromCore": CurveFromCore,
    "CCN_CurveToCore": CurveToCore,
    "CCN_CFGZeroStarScaled": CFGZeroStarScaled,
    "CCN_CroppedImage": CroppedImage,
    "CCN_ImageInset": ImageInset,
    "CCN_HyperRemapKrea2Edit": HyperRemapKrea2Edit,
    "CCN_HyperRemapKrea2EditSlim": HyperRemapKrea2EditSlim,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCN_LoraLoaderByIndex": "LoRA Loader By Index (CCN)",
    "CCN_LoraLoaderFiltered": "LoRA Loader Filtered (CCN)",
    "CCN_ImageLoaderByIndex": "Image Loader By Index (CCN)",
    "CCN_VideoLoaderByIndex": "Video Loader By Index (CCN)",
    "CCN_VideoScrubber": "Video Scrubber (CCN)",
    "CCN_ConditioningNormalizer": "Conditioning Normalizer (CCN)",
    "CCN_ConditioningScale": "Conditioning Scale (CCN)",
    "CCN_ConditioningClamp": "Conditioning Clamp (CCN)",
    "CCN_ConditioningStats": "Conditioning Stats (CCN)",
    "CCN_ConditioningLerp": "Conditioning Lerp (CCN)",
    "CCN_ConditioningSubtract": "Conditioning Subtract (CCN)",
    "CCN_LoraListDirectory": "LoRA List Directory (CCN)",
    "CCN_StringConcatenate": "String Concatenate (CCN)",
    "CCN_StringMergeUnique": "String Merge Unique (CCN)",
    "CCN_StringReplacer": "String Replacer (CCN)",
    "CCN_StringExtractor": "String Extractor (CCN)",
    "CCN_StringListSlicer": "String List Slicer (CCN)",
    "CCN_StringSplitter": "String Splitter (CCN)",
    "CCN_Print": "Print (CCN)",
    "CCN_LatentClamp": "Latent Clamp (CCN)",
    "CCN_LatentScale": "Latent Scale (CCN)",
    "CCN_LatentNormalize": "Latent Normalize (CCN)",
    "CCN_LatentStats": "Latent Stats (CCN)",
    "CCN_LatentChannelOffset": "Latent Channel Offset (CCN)",
    "CCN_LatentChannelOffset16": "Latent Channel Offset x16 (CCN)",
    "CCN_LatentChannelScale": "Latent Channel Scale (CCN)",
    "CCN_LatentChannelScale16": "Latent Channel Scale x16 (CCN)",
    "CCN_PromptBuilder": "Prompt Builder (CCN)",
    "CCN_PromptBuilderB": "Prompt Builder B (CCN)",
    "CCN_PromptStore": "Prompt Store (CCN)",
    "CCN_PromptStoreB": "Prompt Store B (CCN)",
    "CCN_PromptStoreClear": "Prompt Store Clear (CCN)",
    "CCN_PromptStoreCustom": "Prompt Store Custom (CCN)",
    "CCN_PromptStoreGet": "Prompt Store Get (CCN)",
    "CCN_PromptStoreHeadings": "Prompt Store Headings (CCN)",
    "CCN_PromptStoreList": "Prompt Store List (CCN)",
    "CCN_EmphasisEncode": "Emphasis Encode [EXPERIMENTAL] (CCN)",
    "CCN_EmphasisEncodeAdvanced": "Emphasis Encode Advanced [EXPERIMENTAL] (CCN)",
    "CCN_InspectTensor": "Inspect Tensor (CCN)",
    "CCN_RandomSelect": "Random Select (CCN)",
    "CCN_ResizeByShorterEdge": "Resize By Shorter Edge (CCN)",
    "CCN_ResizeToMatch": "Resize To Match (CCN)",
    "CCN_ImageBlend": "Image Blend (CCN)",
    "CCN_LoadJSONFile": "Load JSON File (CCN)",
    "CCN_LoadJSONFilePath": "Load JSON File Path (CCN)",
    "CCN_TokenCounter": "Token Counter (CCN)",
    "CCN_ConditioningTokenCount": "Conditioning Token Count (CCN)",
    "CCN_TokenRemap": "Token Remap (CCN)",
    "CCN_ClipRemap": "CLIP Remap (CCN)",
    "CCN_TokenInspector": "Token Inspector (CCN)",
    "CCN_ConceptRemap": "Concept Remap (CCN)",
    "CCN_HyperRemap": "Hyper Remap (CCN)",
    "CCN_HyperRemapSlim": "Hyper Remap Slim (CCN)",
    "CCN_ConditioningProjectionRemoval": "Conditioning Projection Removal (CCN)",
    "CCN_NeutralPrompt": "Neutral Prompt (CCN)",
    "CCN_NeutralPromptEntry": "Neutral Prompt Entry (CCN)",
    "CCN_NeutralPromptEmpty": "Neutral Prompt Empty (CCN)",
    "CCN_NeutralPromptGuider": "Neutral Prompt Guider (CCN)",
    "CCN_CompoundPrompt": "Compound Prompt (CCN)",
    "CCN_LoraScaleSave": "LoRA Scale & Save (CCN)",
    "CCN_LoraTruncateRank": "LoRA Truncate Rank (CCN)",
    "CCN_LoraMetadata": "LoRA Metadata (CCN)",
    "CCN_SafetensorsMetadata": "Safetensors Metadata (CCN)",
    "CCN_DimensionScale": "Dimension Scale (CCN)",
    "CCN_FloatLerp": "Float Lerp (CCN)",
    "CCN_CurveSample": "Curve Sample (CCN)",
    "CCN_CurveDefinition": "Curve (CCN)",
    "CCN_CurveCFGGuider": "Curve CFG Guider (CCN)",
    "CCN_CurveFromCore": "Curve From Core (CCN)",
    "CCN_CurveToCore": "Curve To Core (CCN)",
    "CCN_CFGZeroStarScaled": "CFG-Zero* Scaled (CCN)",
    "CCN_CroppedImage": "Cropped Image (CCN)",
    "CCN_ImageInset": "Image Inset (CCN)",
    "CCN_HyperRemapKrea2Edit": "Hyper Remap Krea2 Edit (CCN)",
    "CCN_HyperRemapKrea2EditSlim": "Hyper Remap Krea2 Edit Slim (CCN)",
}
