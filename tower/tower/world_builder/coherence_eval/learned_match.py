"""A sub-pixel learned matcher for the coherence experiments: ALIKED + LightGlue.

Why not EfficientLoFTR: the transformers 5.16 port returns its matches on the
8-px coarse grid -- the fine refinement does effectively nothing (review V4b,
median error 3.9-4.5 px on any shift that is not a multiple of 8). Every
conclusion drawn with it described that defect, not the method.

ALIKED is a detector: an image gives the same keypoints in every pair it is
matched in, positions sub-pixel (soft detection). LightGlue matches the
descriptors. Licences: LightGlue code and its ALIKED weights Apache-2.0
(cvg/LightGlue); ALIKED code and weights BSD-3-Clause (Shiaoming/ALIKED). No
SuperPoint anywhere (its weights are non-commercial).

Neither package is in the Tower venv and nothing in the product imports this
module. Experiments put ``lightglue`` (and kornia, which it imports) on
``sys.path``; weights come from the torch hub cache (set TORCH_HOME).

Keypoints are returned in the input image's pixel coordinates, OpenCV
convention (pixel centres at integers) -- the convention of
``EloftrMatcher.match``, so either matcher fits the same callers.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

MATCHER_ID = "aliked-n16+lightglue"
ALIKED_LG_PARAMS = {
    "matcher": MATCHER_ID,
    "licences": "LightGlue Apache-2.0 (cvg/LightGlue), ALIKED BSD-3-Clause (Shiaoming/ALIKED)",
    "extractor": "aliked-n16",
    "max_num_keypoints": 2048,
    "detection_threshold": 0.2,
    # Native resolution: no resize, so no scale mapping to get wrong.
    "resize": None,
    "filter_threshold": 0.1,
    # Adaptive depth/width pruning off: every pair runs the full network, so
    # a pair's matches do not depend on how confident an early layer was.
    "depth_confidence": -1,
    "width_confidence": -1,
    "keypoints": "float, OpenCV convention (pixel centres at integers)",
}


class AlikedLightGlue:
    """``match(rgb_a, rgb_b) -> (kp_a (N,2) f64, kp_b (N,2) f64, score (N,) f32)``.

    Features are cached per image key (``features(rgb, key=...)``), so a
    chain of pairs extracts each image once.
    """

    def __init__(self, device: str = "cuda", params: dict | None = None, cache_size: int = 256):
        import torch
        from lightglue import ALIKED, LightGlue

        self.params = dict(ALIKED_LG_PARAMS, **(params or {}))
        p = self.params
        self.device = device
        self.extractor = ALIKED(model_name=p["extractor"], max_num_keypoints=p["max_num_keypoints"],
                                detection_threshold=p["detection_threshold"]).eval().to(device)
        self.matcher = LightGlue(features="aliked", filter_threshold=p["filter_threshold"],
                                 depth_confidence=p["depth_confidence"],
                                 width_confidence=p["width_confidence"]).eval().to(device)
        self._torch = torch
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = int(cache_size)

    def features(self, rgb, key=None) -> dict:
        """ALIKED features of an HxWx3 uint8 RGB image (on the device)."""
        if key is not None and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        torch = self._torch
        img = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0).to(self.device)
        with torch.inference_mode():
            feats = self.extractor.extract(img, resize=self.params["resize"])
        if key is not None:
            self._cache[key] = feats
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return feats

    def match_features(self, fa: dict, fb: dict):
        """(idx (M,2) int64 into each image's keypoints, score (M,) f32)."""
        torch = self._torch
        with torch.inference_mode():
            out = self.matcher({"image0": fa, "image1": fb})
        m = out["matches"][0].cpu().numpy().astype(np.int64)
        s = out["scores"][0].float().cpu().numpy().astype(np.float32)
        return m.reshape(-1, 2), s.reshape(-1)

    @staticmethod
    def keypoints(feats) -> np.ndarray:
        return feats["keypoints"][0].double().cpu().numpy()

    def match(self, image_a_rgb, image_b_rgb, key_a=None, key_b=None):
        fa = self.features(image_a_rgb, key_a)
        fb = self.features(image_b_rgb, key_b)
        m, s = self.match_features(fa, fb)
        return self.keypoints(fa)[m[:, 0]], self.keypoints(fb)[m[:, 1]], s
