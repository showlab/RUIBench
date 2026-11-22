"""The models behind the vision service: SAM 3, Grounding DINO or OWLv2 with SAM 2.1, and
monocular depth, each at a pinned revision."""

from __future__ import annotations

#: the weights each name stood for when the condition was fixed: a model hub's
#: "main" moves, and the answers with it
REVISIONS = {
    "facebook/sam3": "3c879f39826c281e95690f02c7821c4de09afae7",
    "IDEA-Research/grounding-dino-base": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
    "facebook/sam2.1-hiera-large": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
    "google/owlv2-large-patch14-ensemble": "95e26936e865f87db1742128404b3c035d47d89d",
    "IDEA-Research/grounding-dino-tiny": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf": "8078d68a9c75a972131914f6afd0c1723be0da7f",
}


class Sam3:
    """SAM 3: concepts by phrase (``Sam3Model``) and one thing by a point or a
    box (``Sam3TrackerModel``), from one checkpoint."""

    name = "sam3"
    MODEL = "facebook/sam3"
    #: an instance counts from this score; low, for recall -- what is not a
    #: thing on the table is turned away later, with depth, by the caller, and
    #: the score goes back to the agent.  On the 100 marked frames: at 0.2 coins
    #: 59 of 75 and X pieces ("x piece") 87 of 100, at 0.3 53 and 74; at 0.15
    #: other things begin to answer (a "wooden base": 29 of them)
    THRESHOLD = 0.20

    def __init__(self, device: str) -> None:
        import torch
        from transformers import (
            Sam3Model,
            Sam3Processor,
            Sam3TrackerModel,
            Sam3TrackerProcessor,
        )

        self.torch, self.device = torch, device
        at = REVISIONS[self.MODEL]
        self.proc = Sam3Processor.from_pretrained(self.MODEL, revision=at)
        self.model = Sam3Model.from_pretrained(self.MODEL, revision=at).to(device).eval()
        self.point_proc = Sam3TrackerProcessor.from_pretrained(self.MODEL, revision=at)
        self.point_model = Sam3TrackerModel.from_pretrained(self.MODEL, revision=at).to(device).eval()

    def info(self) -> dict:
        return {"backend": self.name, "detector": self.MODEL, "segmenter": self.MODEL,
                "detector_revision": REVISIONS[self.MODEL],
                "segmenter_revision": REVISIONS[self.MODEL], "threshold": self.THRESHOLD}

    def detect(self, image, phrases):
        """[(phrase, score, box, mask)] for every phrase, the picture encoded once."""
        torch = self.torch
        first = self.proc(images=image, text=phrases[0].strip().lower(), return_tensors="pt").to(self.device)
        with torch.no_grad():
            seen = self.model.get_vision_features(pixel_values=first["pixel_values"])
        out = []
        for phrase in phrases:
            words = self.proc(text=phrase.strip().lower(), return_tensors="pt").to(self.device)
            with torch.no_grad():
                found = self.model(vision_embeds=seen, input_ids=words["input_ids"],
                                   attention_mask=words["attention_mask"])
            res = self.proc.post_process_instance_segmentation(
                found, threshold=self.THRESHOLD, mask_threshold=0.5,
                target_sizes=[image.size[::-1]])[0]
            for mask, box, score in zip(res["masks"], res["boxes"].tolist(), res["scores"].tolist()):
                out.append((phrase, float(score), [float(v) for v in box],
                            mask.cpu().numpy().astype(bool), float(score)))
        return out

    def segment(self, image, point=None, box=None):
        return _prompted(self.torch, self.device, self.point_proc, self.point_model, image,
                         boxes=[box] if box is not None else None, point=point)[0]


class GroundedSam2:
    """Grounding DINO for the boxes, SAM 2.1 for the masks."""

    name = "grounded-sam2"
    DETECTOR = "IDEA-Research/grounding-dino-base"
    SEGMENTER = "facebook/sam2.1-hiera-large"
    #: a box counts from this score, and its phrase from this one
    BOX_THRESHOLD, TEXT_THRESHOLD = 0.25, 0.20

    def __init__(self, device: str) -> None:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
        )

        self.torch, self.device = torch, device
        det_at, seg_at = REVISIONS[self.DETECTOR], REVISIONS[self.SEGMENTER]
        self.det_proc = AutoProcessor.from_pretrained(self.DETECTOR, revision=det_at)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.DETECTOR, revision=det_at).to(device).eval()
        self.seg_proc = Sam2Processor.from_pretrained(self.SEGMENTER, revision=seg_at)
        self.seg = Sam2Model.from_pretrained(self.SEGMENTER, revision=seg_at).to(device).eval()

    def info(self) -> dict:
        return {"backend": self.name, "detector": self.DETECTOR, "segmenter": self.SEGMENTER,
                "detector_revision": REVISIONS[self.DETECTOR],
                "segmenter_revision": REVISIONS[self.SEGMENTER],
                "box_threshold": self.BOX_THRESHOLD, "text_threshold": self.TEXT_THRESHOLD}

    def detect(self, image, phrases):
        torch = self.torch
        boxes = []
        for phrase in phrases:
            inputs = self.det_proc(images=image, text=phrase.strip().lower().rstrip(".") + " .",
                                   return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.det(**inputs)
            res = self.det_proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD, target_sizes=[image.size[::-1]])[0]
            boxes += [(phrase, float(score), [float(v) for v in box])
                      for box, score in zip(res["boxes"].tolist(), res["scores"].tolist())]
        if not boxes:
            return []
        masks = _prompted(torch, self.device, self.seg_proc, self.seg, image,
                          boxes=[b for _, _, b in boxes])
        return [(phrase, score, box, mask, mask_score)
                for (phrase, score, box), (mask, mask_score) in zip(boxes, masks)]

    def segment(self, image, point=None, box=None):
        return _prompted(self.torch, self.device, self.seg_proc, self.seg, image,
                         boxes=[box] if box is not None else None, point=point)[0]


def _prompted(torch, device, proc, model, image, boxes=None, point=None):
    """One mask a box (or one for the point): (bool HxW array, score) each."""
    if boxes is not None:
        inputs = proc(images=image, input_boxes=[[list(b) for b in boxes]],
                      return_tensors="pt").to(device)
    else:
        inputs = proc(images=image, input_points=[[[list(point)]]], input_labels=[[[1]]],
                      return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=boxes is None)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
    scores = out.iou_scores.cpu()[0]
    found = []
    for k in range(masks.shape[0]):
        best = int(scores[k].argmax())
        found.append((masks[k, best].numpy().astype(bool), float(scores[k, best])))
    return found


class _BoxesThenSam2:
    """A box detector for the words, then SAM 2.1 for the masks: the shape of
    ``gdino-tiny-sam2`` and ``owlv2-sam2``, where the detector is the only
    difference."""

    name = ""
    DETECTOR = ""
    SEGMENTER = "facebook/sam2.1-hiera-large"
    #: a box counts from this score (the detector's own scale)
    BOX_THRESHOLD = 0.25

    def __init__(self, device: str) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch, self.device = torch, device
        seg_at = REVISIONS[self.SEGMENTER]
        self.seg_proc = Sam2Processor.from_pretrained(self.SEGMENTER, revision=seg_at)
        self.seg = Sam2Model.from_pretrained(self.SEGMENTER, revision=seg_at).to(device).eval()
        self._load_detector(REVISIONS[self.DETECTOR])

    def _load_detector(self, revision: str) -> None:
        raise NotImplementedError

    def boxes(self, image, phrases):
        """[(phrase, score, [u0, v0, u1, v1])] for every phrase."""
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.name, "detector": self.DETECTOR, "segmenter": self.SEGMENTER,
                "detector_revision": REVISIONS[self.DETECTOR],
                "segmenter_revision": REVISIONS[self.SEGMENTER],
                "box_threshold": self.BOX_THRESHOLD}

    def detect(self, image, phrases):
        boxes = self.boxes(image, phrases)
        if not boxes:
            return []
        masks = _prompted(self.torch, self.device, self.seg_proc, self.seg, image,
                          boxes=[b for _, _, b in boxes])
        return [(phrase, score, box, mask, mask_score)
                for (phrase, score, box), (mask, mask_score) in zip(boxes, masks)]

    def segment(self, image, point=None, box=None):
        return _prompted(self.torch, self.device, self.seg_proc, self.seg, image,
                         boxes=[box] if box is not None else None, point=point)[0]


class _GroundingStyle(_BoxesThenSam2):
    """Detectors shaped like Grounding DINO: one phrase a pass, scored by the
    phrase's tokens (``post_process_grounded_object_detection``)."""

    TEXT_THRESHOLD = 0.20

    def _load_detector(self, revision: str) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.det_proc = AutoProcessor.from_pretrained(self.DETECTOR, revision=revision)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.DETECTOR, revision=revision).to(self.device).eval()

    def info(self) -> dict:
        return {**super().info(), "text_threshold": self.TEXT_THRESHOLD}

    def boxes(self, image, phrases):
        out = []
        for phrase in phrases:
            inputs = self.det_proc(images=image, text=phrase.strip().lower().rstrip(".") + " .",
                                   return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                raw = self.det(**inputs)
            res = self.det_proc.post_process_grounded_object_detection(
                raw, inputs.input_ids, threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD, target_sizes=[image.size[::-1]])[0]
            out += [(phrase, float(score), [float(v) for v in box])
                    for box, score in zip(res["boxes"].tolist(), res["scores"].tolist())]
        return out


class GdinoTinySam2(_GroundingStyle):
    """Grounding DINO tiny and SAM 2.1: ``grounded-sam2``'s pair with a smaller detector."""

    name = "gdino-tiny-sam2"
    DETECTOR = "IDEA-Research/grounding-dino-tiny"
    BOX_THRESHOLD, TEXT_THRESHOLD = 0.25, 0.20


class OwlSam2(_BoxesThenSam2):
    """OWLv2 (large, ensemble) and SAM 2.1.  OWLv2 scores every box against
    every phrase at once, each on its own (a sigmoid, not a share), so the
    phrases still never merge; its scores run low -- 0.1 is a find."""

    name = "owlv2-sam2"
    DETECTOR = "google/owlv2-large-patch14-ensemble"
    BOX_THRESHOLD = 0.10

    def _load_detector(self, revision: str) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.det_proc = Owlv2Processor.from_pretrained(self.DETECTOR, revision=revision)
        self.det = Owlv2ForObjectDetection.from_pretrained(
            self.DETECTOR, revision=revision).to(self.device).eval()

    def boxes(self, image, phrases):
        words = [p.strip().lower() for p in phrases]
        inputs = self.det_proc(text=[words], images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            raw = self.det(**inputs)
        # OWLv2 pads the picture to a square (right and bottom) before it looks
        side = max(image.size)
        res = self.det_proc.post_process_grounded_object_detection(
            raw, threshold=self.BOX_THRESHOLD, target_sizes=[(side, side)],
            text_labels=[words])[0]
        width, height = image.size
        out = []
        for box, score, k in zip(res["boxes"].tolist(), res["scores"].tolist(),
                                 res["labels"].tolist()):
            box = [max(0.0, box[0]), max(0.0, box[1]), min(float(width), box[2]),
                   min(float(height), box[3])]
            if box[2] - box[0] >= 1 and box[3] - box[1] >= 1:
                out.append((phrases[int(k)], float(score), box))
        return out


class MonoDepth:
    """Depth from the RGB picture alone: Depth Anything V2, metric, trained on
    indoor scenes (small: Apache-2.0).  Its depth is in metres along the optical
    axis, scaled as the model learnt; the client fits the scale to the table top
    the robot knows."""

    name = "dav2-small"
    MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

    def __init__(self, device: str) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch, self.device = torch, device
        at = REVISIONS[self.MODEL]
        self.proc = AutoImageProcessor.from_pretrained(self.MODEL, revision=at)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.MODEL, revision=at).to(device).eval()

    def info(self) -> dict:
        return {"depth_model": self.MODEL, "depth_revision": REVISIONS[self.MODEL]}

    def depth(self, image):
        inputs = self.proc(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        pred = self.proc.post_process_depth_estimation(out, target_sizes=[image.size[::-1]])[0]
        return pred["predicted_depth"].float().cpu().numpy()


DEPTHS = {"dav2-small": MonoDepth}


BACKENDS = {"sam3": Sam3, "grounded-sam2": GroundedSam2, "gdino-tiny-sam2": GdinoTinySam2,
            "owlv2-sam2": OwlSam2}
