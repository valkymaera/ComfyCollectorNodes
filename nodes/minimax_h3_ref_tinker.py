"""MiniMax H3 Ref Tinker.

Experimental control over H3 reference conditioning (ref2va), exposing the
levers identified while investigating why reference images behave like start
frames:

- In the packed sequence [text | refs | audio | video], image refs are cond
  rows re-injected near-clean every step, positioned 1.0 RoPE tick before the
  target video (1 tick = 1/40 s on the shared AV clock; video frame hops are
  1.67 / 6.67 ticks). Kinds "cond" (fl2va keyframes) and "ref_img" are
  otherwise identical to the model - position and prompt presentation are the
  only role signals.
- model_base.MiniMaxH3.extra_conds already forwards two unexposed payload
  knobs: minimax_visual_cond_noise_aug / minimax_audio_cond_noise_aug.

This node patches the diffusion-model call (patcher_extension wrapper) and
rewrites a per-call copy of the minimax payload:

- visual/audio cond noise aug: reference-strength. The model mixes
  aug * latent + (1 - aug) * fixed-seed noise into cond rows and pins their
  declared timestep at max(t, aug). Stock: 0.999 visual / 1.0 audio.
- ref_temporal_gap: extra RoPE-time distance inserted between reference
  blocks and the target AV streams (keyframe anchors stay glued to their
  frames). -1.0 co-locates the last ref with frame 0 (soft keyframe mode).
- ref_spatial_shift_h/w: offset added to image-ref RoPE h/w coordinates
  (frame grids are area-normalized, spanning roughly 0-32) to decorrelate
  ref<->target pixel alignment without editing the image.

Registered through the CCN package __init__. The node sits in
model/patch/minimax next to the official H3 nodes and composes with other
model patches (sigma shift, attention mods, sampler packs).
"""

import copy

import comfy.patcher_extension
from comfy.ldm.minimax.model import PackedLayout

WRAPPER_KEY = "minimax_h3_ref_tinker"
_HW_MULT = 2  # H3 DiT patch (1, 2, 2): the layout rounds latent h/w up to 2


def _resolve_layout(payload, x, context):
    """Return the PackedLayout the forward pass would use for this call.

    Mirrors model.py's resolution: the payload's prebuilt layout when its
    signature matches the (padded) latent shapes, else a stock rebuild.
    """
    video_x, audio_x = x[0], x[1]
    lat_h = (video_x.shape[3] + _HW_MULT - 1) // _HW_MULT * _HW_MULT
    lat_w = (video_x.shape[4] + _HW_MULT - 1) // _HW_MULT * _HW_MULT
    signature = (context.shape[1], video_x.shape[2], lat_h, lat_w, audio_x.shape[-1])
    layout = payload.get("layout")
    if layout is not None and layout.signature == signature:
        return layout
    return PackedLayout(*signature, keyframes=payload.get("keyframes"),
                        refs=payload.get("refs"), frame_count=payload.get("frame_count"))


def _make_wrapper(vis_aug, aud_aug, gap, shift_h, shift_w):
    move_refs = gap != 0.0 or shift_h != 0.0 or shift_w != 0.0

    def tinker_wrapper(executor, x, timestep, context, transformer_options={},
                       minimax_payload=None, **kwargs):
        payload = minimax_payload
        if isinstance(payload, dict):
            # Work on copies only: the payload dict and layout are built once
            # per sampling run and shared across every step / conditioning.
            payload = dict(payload)
            payload["visual_cond_noise_aug"] = vis_aug
            payload["audio_cond_noise_aug"] = aud_aug
            if move_refs and payload.get("refs"):
                layout = _resolve_layout(payload, x, context)
                pos = layout.position_ids.clone()
                for a, b, kind in layout.segments:
                    if kind == "ref_img":
                        pos[a:b, 1] += shift_h
                        pos[a:b, 2] += shift_w
                    elif kind in ("cond", "audio", "video"):
                        # Widen the ref->target gap by pushing the target AV
                        # streams forward; "cond" keyframe anchors move with
                        # the frames they are glued to.
                        pos[a:b, 0] += gap
                layout = copy.copy(layout)  # shallow: only position_ids is replaced
                layout.position_ids = pos
                payload["layout"] = layout
        return executor(x, timestep, context, transformer_options,
                        minimax_payload=payload, **kwargs)

    return tinker_wrapper


class MiniMaxH3RefTinker:
    DESCRIPTION = ("Experimental control over MiniMax H3 reference conditioning: "
                   "reference strength (cond-row noise aug) and reference RoPE "
                   "geometry (temporal gap to the target streams, spatial offset "
                   "of image refs). Wraps the diffusion-model call; composes with "
                   "other model patches.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {
                "default": True,
                "tooltip": "Manual off switch - passes the model through unpatched."}),
            "visual_cond_noise_aug": ("FLOAT", {
                "default": 0.999, "min": 0.0, "max": 1.0, "step": 0.001,
                "tooltip": "Strength of visual condition rows (reference images and fl2va "
                           "keyframes). 0.999 = stock (rows re-injected ~clean every step). "
                           "Lower mixes fixed-seed noise into the rows (aug*latent + "
                           "(1-aug)*noise) and lowers their declared timestep, weakening "
                           "verbatim copying before identity. Useful band ~0.90-1.00."}),
            "audio_cond_noise_aug": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                "tooltip": "Same for reference-audio rows. 1.0 = stock (fully clean)."}),
            "ref_temporal_gap": ("FLOAT", {
                "default": 0.0, "min": -1.0, "max": 2000.0, "step": 0.1,
                "tooltip": "Extra RoPE-time distance between reference blocks and the target "
                           "audio/video streams, in position ticks (1 tick = 1/40 s on the "
                           "shared AV clock). Stock ref->video gap is 1.0 tick; video frame "
                           "hops are 1.67/6.67. -1.0 co-locates the last image ref with "
                           "frame 0 (soft keyframe-anchor mode). 0 = stock."}),
            "ref_spatial_shift_h": ("FLOAT", {
                "default": 0.0, "min": -64.0, "max": 64.0, "step": 0.25,
                "tooltip": "Offset added to image-reference RoPE h coordinates (frame grids "
                           "are area-normalized, roughly spanning 0-32). Decorrelates "
                           "ref<->target pixel alignment without editing the image. 0 = stock."}),
            "ref_spatial_shift_w": ("FLOAT", {
                "default": 0.0, "min": -64.0, "max": 64.0, "step": 0.25,
                "tooltip": "Offset added to image-reference RoPE w coordinates. 0 = stock."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patch/minimax"

    def patch(self, model, enabled, visual_cond_noise_aug, audio_cond_noise_aug,
              ref_temporal_gap, ref_spatial_shift_h, ref_spatial_shift_w):
        if not enabled:
            return (model,)
        m = model.clone()
        wrapper = _make_wrapper(float(visual_cond_noise_aug), float(audio_cond_noise_aug),
                                float(ref_temporal_gap), float(ref_spatial_shift_h),
                                float(ref_spatial_shift_w))
        # Replace-not-stack: write the {"wrapper_type": {"key": [fn]}} structure
        # directly. patcher_extension.add_wrapper_with_key appends to the key's
        # list, which would stack tinker wrappers when a chain re-patches the
        # same model. Copy the levels we touch so the source patcher's
        # model_options are never mutated.
        wt = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
        to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
        wrappers = to["wrappers"] = {k: dict(v) for k, v in to.get("wrappers", {}).items()}
        wrappers.setdefault(wt, {})[WRAPPER_KEY] = [wrapper]
        return (m,)

