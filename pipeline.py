#app.pipeline.py
# -*- coding: utf-8 -*-
"""
Processing pipeline with:
- initial_yolo_rate (YOLO throttle)
- yolo_mode "auto" to backpressure YOLO while OCR busy
- max_task_queue and drop_frames_when_busy to drop frames when queue is full
- heartbeat, stream_mode, push to webserver, etc.

This cleaned version removes internal server-side filtering/aggregation.
The pipeline emits raw OCR results (merged + checks + detections + image_b64)
to the web backend for further processing (filter/gather/logging).
"""
from __future__ import annotations

import os
import inspect

import asyncio
import time
import json
from typing import Any, Optional, List, Tuple, Dict
from threading import Thread, Event, Lock, Condition
from queue import Queue, Empty
from pathlib import Path
import uuid
import hashlib
import cv2
import numpy as np
import gc
from datetime import datetime
import sys
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from PIL import Image
from io import BytesIO
import base64
import logging
import requests
import threading

from app import models
from app.result_manager import ResultManager
from app.filter import FilterManager

from app.utils import (
    check_container_bic,
    check_class2_format,
    safe_filename,
    enhance_gray_for_crnn,
    horizontal_scale_and_pad,
    pil_save_with_footer,
    build_candidates,
    attempt_merged_retry_class1,
    attempt_retry_class2,
    calculate_checksum_custom,
    expand_box,
    build_parts_for_classes,
    make_display_timestamp,
    encode_image_to_base64, clamp_bbox, scale_detections,compute_union_bbox,create_crop_bytes,
)

from app import websocket_manager

# Import the safe sender utility
from app.ws_send_utils import safe_send_webpayload

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# 기존 pipeline_result_tabbed_summary.py 의 out 디렉토리 경로 설정
OUT_DIR = ROOT / "out"
CROPS_DIR = OUT_DIR / "crops"
PREPROCESSED_DIR = OUT_DIR / "preprocessed"
RESULTS_DIR = OUT_DIR / "results"

_MIN_SEPARATOR_LEN = 20
_MAX_SEPARATOR_LEN = 100

_CONTAINER_RE = re.compile(r"^([A-Z]{3}U)(\d{7})$")


class ProcessingPipeline:
    def __init__(self, config: dict):
        self.config = config or {}
        try:
            cam_cfg = self.config.get("camera", {}) or {}
            cam_w = int(cam_cfg.get("width", 0) or 0)
            cam_h = int(cam_cfg.get("height", 0) or 0)
            if cam_w > 0 and cam_h > 0:
                self.orig_size = [cam_w, cam_h]
            else:
                self.orig_size = None
        except Exception:
            self.orig_size = None

        self.yolo_cfg = self.config.get("yolo", {})
        self.crnn_cfg = self.config.get("crnn", {})
        self.yolo_model = None
        self.crnn_model = None
        self._frame_seq = 0
        self._frame_seq_lock = Lock()

        # ROI (None or dict with keys x,y,w,h in image pixel coords)
        self._roi = None   # None or dict {'x':int,'y':int,'w':int,'h':int}
        self._roi_lock = Lock()

        # pending normalized ROI (will be applied on first frame when runtime size known)
        # structure: {"x":float,"y":float,"w":float,"h":float,"normalized":True}
        self._pending_roi = None
        self._roi_applied_once = False

        # processing config
        self.processing_cfg = self.config.get("processing", {}) or {}

        # ========== Config 캐싱 (성능 개선) ==========
        # 이미지/스트리밍 관련
        self.stream_max_width = int(self.processing_cfg.get("stream_max_width", 640) or 640)
        self.stream_jpeg_quality = int(self.processing_cfg.get("stream_jpeg_quality", 70) or 70)
        
        # Union crop 관련
        self.union_crop_target_width = int(self.processing_cfg.get("union_crop_width", 200) or 200)
        self.union_crop_pad = int(self.processing_cfg.get("union_crop_pad", 8) or 8)
        self.union_crop_jpeg_quality = int(self.processing_cfg.get("union_crop_jpeg_quality", 90) or 90)
        
        # Bbox expand 관련
        self.bbox_expand_pad = int(self.processing_cfg.get("bbox_expand_pad", 8) or 8)
        
        # Resolution change 관련
        self.resolution_stable_threshold = int(self.processing_cfg.get("resolution_stable_threshold", 3) or 3)
        # ========== 캐싱 끝 ==========

        # stream options
        self.save_crops = bool(self.processing_cfg.get("save_crops", False))
        self.save_preprocessed = bool(self.processing_cfg.get("save_preprocessed", True))
        self.preprocessed_dir = Path(self.processing_cfg.get("preprocessed_dir", str(Path("out/preprocessed"))))
        self.include_empty = bool(self.processing_cfg.get("include_empty", False))
        self.save_annotated = bool(self.processing_cfg.get("save_annotated", True))

        # stream options: always send frame to webserver even if no detections
        self.always_stream = bool(self.processing_cfg.get("always_stream", True))
        # stream_mode: prefer explicit config, else derive from always_stream for backward compatibility
        cfg_stream_mode = self.processing_cfg.get("stream_mode", None)
        if cfg_stream_mode is not None:
            try:
                sm = str(cfg_stream_mode).strip().lower()
                if sm in ("always", "detect", "stop"):
                    self.stream_mode = sm
                else:
                    self.stream_mode = "detect"
            except Exception:
                self.stream_mode = "detect"
        else:
            self.stream_mode = "always" if self.always_stream else "detect"

        self.stream_interval = float(self.processing_cfg.get("stream_interval", 0.0) or 0.0)
        self._last_stream_ts = 0.0
        if self.always_stream:
            logger.info("always_stream=ON (stream_interval=%.3fs)", self.stream_interval)

        # display/filter options (server-side aggregation)
        # self.display_filter_enabled = bool(self.processing_cfg.get("display_filter_enabled", True))
        self.filter_interval = float(self.processing_cfg.get("filter_interval", 0.0) or 0.0)
        self.display_filter_enabled = True if (self.filter_interval and float(self.filter_interval) > 0.0) else False
        self._recent_results: List[dict] = []
        self._recent_lock = Lock()

        # filtered hold state: store last filtered values and timestamps
        self._filtered_last = {
            "class1": {"val": "", "check": {}, "ts": 0.0},
            "class2": {"val": "", "check": {}, "ts": 0.0},
        }

        # runtime
        self._ws_clients = set()
        self._running = False
        self._worker_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._ready_event = Event()

        self._filter_thread: Optional[Thread] = None
        # NOTE: display_filter_enabled start logic has been moved to after FilterManager
        # is instantiated below. Do not start a legacy _filter_loop here because the
        # implementation may not exist at this point.

        # OCR retry config
        self.ocr_retry_scales = self.processing_cfg.get("ocr_retry_scales", [0.9])
        if not isinstance(self.ocr_retry_scales, (list, tuple)):
            self.ocr_retry_scales = [self.ocr_retry_scales]
        self.ocr_retry_scales = [float(s) for s in self.ocr_retry_scales][:3]

        self.min_ocr_conf = float(self.processing_cfg.get("min_ocr_conf", 0.6))
        self.candidate_list: List[str] = list(self.processing_cfg.get("candidate_list", ["enh", "gray", "orig"]))
        self.candidate_mode: str = str(self.processing_cfg.get("candidate_mode", "all"))
        self.retry_candidate_list: List[str] = list(self.processing_cfg.get("retry_candidate_list", ["enh"]))
        self.retry_candidate_mode: str = str(self.processing_cfg.get("retry_candidate_mode", "first"))

        self.force_retry_enabled = bool(self.processing_cfg.get("force_retry_enabled", True))
        self.debug_retry = bool(self.processing_cfg.get("debug_retry", True))

        self.debug_timing = bool(self.processing_cfg.get("debug_timing", False))
        self.warmup_infer = bool(self.processing_cfg.get("warmup_infer", True))
        self.warmup_yolo_runs = int(self.processing_cfg.get("warmup_yolo_runs", 3))
        self.warmup_crnn_runs = int(self.processing_cfg.get("warmup_crnn_runs", 2))

        # initial YOLO rate (runs per second). 0 or <=0 means no throttling.
        self.initial_yolo_rate = float(self.processing_cfg.get("initial_yolo_rate", 0.0) or 0.0)
        self._last_yolo_ts = 0.0
        if self.initial_yolo_rate > 0:
            logger.info("initial_yolo_rate=%s (interval=%.3fs)", self.initial_yolo_rate, 1.0 / self.initial_yolo_rate)
        else:
            logger.info("initial_yolo_rate=UNLIMITED")

        # yolo_mode: "manual" (use initial_yolo_rate/throttle) or "auto" (backpressure: skip YOLO while OCR busy)
        self.yolo_mode = str(self.processing_cfg.get("yolo_mode", "manual"))

        # queue/backpressure config
        self.max_task_queue = int(self.processing_cfg.get("max_task_queue", 4))
        self.drop_frames_when_busy = bool(self.processing_cfg.get("drop_frames_when_busy", True))

        # heartbeat
        self.heartbeat_interval = float(self.processing_cfg.get("heartbeat_interval", 10.0) or 0.0)
        self._last_heartbeat_ts = 0.0
        if self.heartbeat_interval > 0:
            logger.info("heartbeat_interval=%.1fs", self.heartbeat_interval)

        # time block helper
        if self.debug_timing:
            try:
                from app.pipeline_time_helpers import time_block as _real_time_block
                self._time_block = _real_time_block
            except Exception:
                @contextmanager
                def _noop_tb(name: str):
                    yield
                self._time_block = _noop_tb
        else:
            @contextmanager
            def _noop_tb(name: str):
                yield
            self._time_block = _noop_tb

        # runtime queues / executors
        self._task_queue: "Queue[tuple[str, tuple, Queue]]" = Queue()
        self._cuda_ctx = None
        self._io_executor = ThreadPoolExecutor(max_workers=4)

        # OCR in-flight tracking (for auto backpressure)
        self._ocr_inflight = 0
        self._ocr_lock = Lock()
        self._ocr_cond = Condition(self._ocr_lock)

        # create ResultManager with defaults (no path args required)
        self.result_manager = ResultManager(
            results_dir=None,
            crops_dir=None,
            retention_days=int(self.processing_cfg.get("retention_days", 10) or 10),
            save_crop_policy=self.processing_cfg.get("save_crop_policy", {"class1": ["NG", "INVALID"], "class2": ["NG"]}),
            socket_config=self.processing_cfg.get("socket_config", {}) or {},
            web_config=self.processing_cfg.get("web_config", {}) or {},
            enable_per_result_json=bool(self.processing_cfg.get("enable_per_result_json", False)),
            enable_per_result_annotated=bool(self.processing_cfg.get("enable_per_result_annotated", False)),
            crops_executor_workers=int(self.processing_cfg.get("crops_executor_workers", 2) or 2),
        )

        # Immediately apply processing/web config (reconfigure will be defensive)
        try:
            proc_cfg = self.processing_cfg or {}
            web_subset = {
                "webserver_url": proc_cfg.get("webserver_url") or (proc_cfg.get("web") or {}).get("webserver_url"),
                "timeout": proc_cfg.get("web_timeout") or (proc_cfg.get("web") or {}).get("timeout"),
                "frame_store_dir": proc_cfg.get("frame_store_dir"),
                "frame_retention_days": proc_cfg.get("frame_retention_days"),
                "min_crop_area_pixels": proc_cfg.get("min_crop_area_pixels"),
                "ftp_upload": (proc_cfg.get("ftp_upload") or (proc_cfg.get("web") or {}).get("ftp_upload")) or {},
            }
            self.result_manager.reconfigure(processing_cfg=proc_cfg, web_config=web_subset)
        except Exception:
            logger.exception("Failed to apply processing/web config to ResultManager at pipeline init")

        # Hook for FilterManager emit -> ResultManager.write
        def _on_filtered_emit(filtered_payload: dict):
            """
            Emit callback invoked by FilterManager when an aggregated payload is ready.
            Behavior:
            - log a short summary
            - if payload misses image_b64 or detections, try to enrich from recent raw events (self._recent_results)
            - call self.result_manager.write(enriched_payload)
            """
            try:
                fp = filtered_payload or {}
                # Debug summary
                try:
                    logger.debug("Filtered emit: keys=%s merged=%s checks=%s",
                                 list(fp.keys()), fp.get("merged"), fp.get("checks"))
                    dets = fp.get("detections") or fp.get("scaled_detections") or []
                    logger.debug("  detections_count=%d", len(dets))
                    if dets:
                        first = dets[0]
                        logger.debug("    sample det: class=%s bbox=%s ocr_text=%s",
                                     first.get("class"), first.get("bbox") or first.get("box") or first.get("xywh"),
                                     first.get("ocr_text") or first.get("text"))
                    logger.debug("  image_present=%s", bool(fp.get("image_b64") or fp.get("frame_b64")))
                except Exception:
                    logger.exception("Failed logging filtered_payload before write")

                # Enrichment: if missing image or detections, try to find recent candidate
                try:
                    need_image = not bool(fp.get("image_b64") or fp.get("frame_b64"))
                    need_dets = not bool(fp.get("detections"))

                    if need_image or need_dets:
                        cand = None
                        uid = fp.get("uid")
                        with self._recent_lock:
                            # Prefer matching uid
                            if uid and uid != "filtered":
                                for r in reversed(self._recent_results):
                                    if r.get("uid") == uid:
                                        cand = r
                                        break
                            # Fallback: prefer most recent with detections (and image if required)
                            if cand is None:
                                for r in reversed(self._recent_results):
                                    has_dets = bool(r.get("detections"))
                                    has_img = bool(r.get("image_b64"))
                                    if need_dets and has_dets:
                                        cand = r
                                        break
                                    if need_image and has_img and (not need_dets):
                                        cand = r
                                        break
                                # Final fallback: take most recent record if any
                                if cand is None and self._recent_results:
                                    cand = self._recent_results[-1]

                        if cand:
                            logger.debug("_on_filtered_emit: enriching from recent uid=%s ts=%s dets=%d img=%s",
                                         cand.get("uid"), cand.get("ts"), len(cand.get("detections") or []), bool(cand.get("image_b64")))
                            if need_image:
                                if cand.get("image_b64"):
                                    fp["image_b64"] = cand.get("image_b64")
                                elif cand.get("frame_b64"):
                                    fp["frame_b64"] = cand.get("frame_b64")
                            if need_dets and cand.get("detections"):
                                fp["detections"] = cand.get("detections") or []
                        else:
                            logger.debug("_on_filtered_emit: no candidate to enrich payload (need_image=%s need_dets=%s). recent_count=%d",
                                         need_image, need_dets, len(self._recent_results))
                    # final debug of what will be written
                    logger.debug("Prepared payload for write: keys=%s merged=%s checks=%s detections_count=%d image_present=%s",
                                 list(fp.keys()), fp.get("merged"), fp.get("checks"),
                                 len(fp.get("detections") or []), bool(fp.get("image_b64") or fp.get("frame_b64")))
                except Exception:
                    logger.exception("Failed to prepare filtered_payload (fallback enrichment)")

                # Write via ResultManager (non-blocking network ops inside)
                try:
                    saved = self.result_manager.write(fp)
                    logger.debug("ResultManager.write returned: %s", saved)
                except Exception:
                    logger.exception("ResultManager.write failed in emit_fn")
            except Exception:
                logger.exception("Unexpected error in _on_filtered_emit")

        # Instantiate FilterManager only if configured (filter_interval > 0)
        try:
            filter_interval_cfg = float(self.processing_cfg.get("filter_interval", 0.0) or 0.0)
            sim_threshold_cfg = float(self.processing_cfg.get("sim_threshold", 0.20) or 0.20)
            filter_suppress_cfg = float(self.processing_cfg.get("filter_suppress_seconds", 30.0) or 30.0)

            if filter_interval_cfg > 0.0:
                self.filter_manager = FilterManager(
                    emit_fn=_on_filtered_emit,
                    interval=filter_interval_cfg,
                    sim_threshold=sim_threshold_cfg,
                    suppress_repeat_seconds=filter_suppress_cfg
                )
                # optional debug flag
                try:
                    self.filter_manager.debug = bool(self.processing_cfg.get("filter_debug", False))
                except Exception:
                    self.filter_manager.debug = False

                # start filter manager
                try:
                    self.filter_manager.start()
                    logger.info("Display filter enabled: started FilterManager (interval=%.3f)", filter_interval_cfg)
                except Exception:
                    logger.exception("Failed to start FilterManager")
            else:
                self.filter_manager = None
                logger.info("FilterManager disabled (filter_interval=%s)", filter_interval_cfg)
        except Exception:
            self.filter_manager = None
            logger.exception("Failed to initialize FilterManager/ResultManager integration")

    def _resize_for_streaming(self, img: Optional[np.ndarray], max_width: Optional[int] = None) -> Optional[np.ndarray]:
        """Resize image for streaming if width exceeds max_width."""
        if img is None:
            return None
        max_w = max_width or self.stream_max_width
        h, w = img.shape[:2]
        if w <= max_w:
            return img
        scale = max_w / float(w)
        new_w = max_w
        new_h = max(1, int(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    

    # ========== Payload Building ==========
    
    def _build_clients_payload(self, uid: str, ts: float, frame_id: int,
                               orig_size: tuple, sent_size: tuple,
                               detections: list, merged: dict, checks: dict,
                               image_b64: str = "", image_data_uri: str = "",
                               roi: Optional[dict] = None) -> dict:
        """Build clients payload."""
        return {
            "timestamp": ts, "uid": uid, "frame_id": frame_id, "source": "realtime",
            "orig_size": list(orig_size), "sent_size": list(sent_size),
            "detections": detections, "merged": merged, "checks": checks,
            "image_b64": image_b64, "image_data_uri": image_data_uri, "roi": roi,
            "filtered": False, "filter_interval": float(self.filter_interval),
        }
    
    def _build_internal_payload(self, uid: str, ts: float, frame_id: int,
                                orig_size: tuple, sent_size: tuple,
                                detections_full: list, merged: dict, checks: dict,
                                frame_b64: str = "", roi: Optional[dict] = None) -> dict:
        """Build internal payload."""
        return {
            "timestamp": ts, "uid": uid, "frame_id": frame_id, "source": "realtime",
            "orig_size": list(orig_size), "sent_size": list(sent_size),
            "detections": detections_full, "detections_full": detections_full,
            "merged": merged, "checks": checks, "frame_b64": frame_b64, "roi": roi,
            "filtered": False, "filter_interval": float(self.filter_interval),
        }
    '''
    def _create_union_crop_bytes(self, frame: np.ndarray, bbox: tuple) -> Optional[bytes]:
        """Create union crop JPEG bytes with padding and resizing."""
        return create_crop_bytes(
                frame, bbox,
                pad=self.union_crop_pad,
                target_width=self.union_crop_target_width,
                jpeg_quality=self.union_crop_jpeg_quality
            )
    '''

    def load_models(self):
        try:
            self.yolo_model = models.load_yolo_model(self.yolo_cfg)
            logger.info("YOLO model loaded")
        except Exception as e:
            logger.exception("Failed to load YOLO model: %s", e)
            self.yolo_model = None
        try:
            self.crnn_model = models.load_crnn_model(self.crnn_cfg)
            logger.info("CRNN model loaded")
        except Exception as e:
            logger.exception("Failed to load CRNN model: %s", e)
            self.crnn_model = None

        if self.warmup_infer:
            try:
                if self.yolo_model is not None:
                    dummy_size = int(self.yolo_cfg.get("imgsz", 640))
                    dummy = np.full((dummy_size, dummy_size, 3), 127, dtype=np.uint8)
                    for _ in range(max(1, self.warmup_yolo_runs)):
                        try:
                            _ = models.yolo_infer(self.yolo_model, dummy,
                                                  input_size=self.yolo_cfg.get("imgsz", 640),
                                                  conf_thres=0.01, iou_thres=0.5)
                        except Exception:
                            pass
                if self.crnn_model is not None:
                    pil = Image.fromarray(np.full((32, 160), 255, dtype=np.uint8)).convert('L')
                    for _ in range(max(1, self.warmup_crnn_runs)):
                        try:
                            _ = models.crnn_infer(self.crnn_model, pil)
                        except Exception:
                            pass
                logger.info("Warmup finished")
            except Exception as e:
                logger.exception("Warmup failed: %s", e)

        try:
            if self.yolo_model is not None or self.crnn_model is not None:
                self._ready_event.set()
                logger.info("Pipeline ready (models loaded/warmup complete)")
        except Exception:
            pass

    def set_roi(self, x: int, y: int, w: int, h: int):
        """
        Set ROI in pixel coords. If x is None (or x is falsy and clear semantics), clear ROI.
        Return (ok:bool, message:str)
        This version prefers runtime frame size (self.orig_size or self.last_frame_size)
        for clamping. Falls back to config.camera if runtime size not available.
        """
        try:
            with self._roi_lock:
                if x is None:
                    self._roi = None
                    logger.info("ROI cleared via set_roi")
                    return True, "roi cleared"

                # ensure ints
                xi = int(x); yi = int(y); wi = int(w); hi = int(h)
                cam_w = 0; cam_h = 0
                try:
                    # pipeline may expose the last received/orig frame size
                    if hasattr(self, "orig_size") and self.orig_size:
                        cand = self.orig_size
                        cam_w = int(cand[0]); cam_h = int(cand[1])
                    elif hasattr(self, "last_frame_size") and self.last_frame_size:
                        cand = self.last_frame_size
                        cam_w = int(cand[0]); cam_h = int(cand[1])
                except Exception:
                    logger.exception("Failed to read pipeline runtime frame size, will fallback to config")

                # fallback to configured camera size
                if not cam_w or not cam_h:
                    cam = self.config.get("camera", {}) or {}
                    try:
                        cam_w = int(cam.get("width", 0) or 0)
                        cam_h = int(cam.get("height", 0) or 0)
                    except Exception:
                        cam_w = 0; cam_h = 0

                # perform clamping if we have valid cam size
                if cam_w > 0 and cam_h > 0:
                    xi = max(0, min(xi, cam_w - 1))
                    yi = max(0, min(yi, cam_h - 1))
                    wi = max(1, min(wi, cam_w - xi))
                    hi = max(1, min(hi, cam_h - yi))

                # store
                self._roi = {'x': xi, 'y': yi, 'w': wi, 'h': hi}
            logger.info("ROI set to %s", self._roi)
            return True, "roi set"
        except Exception as e:
            logger.exception("set_roi error: %s", e)
            return False, str(e)

    def get_roi(self):
        with self._roi_lock:
            return dict(self._roi) if self._roi is not None else None

    def ensure_output_dirs(self):
        global CROPS_DIR, RESULTS_DIR, PREPROCESSED_DIR
        out_dirs = self.processing_cfg.get("output_dirs", {})
        crops_dir = Path(out_dirs.get("crops", CROPS_DIR))
        results_dir = Path(out_dirs.get("results", RESULTS_DIR))
        preproc_dir = Path(out_dirs.get("preprocessed", str(self.preprocessed_dir)))
        crops_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        preproc_dir.mkdir(parents=True, exist_ok=True)
        CROPS_DIR = crops_dir
        RESULTS_DIR = results_dir
        self.preprocessed_dir = preproc_dir

    def start_background_loop(self):
        if self._running:
            return
        try:
            self._ready_event.clear()
        except Exception:
            pass
        self._stop_event.clear()
        self._running = True
        self._worker_thread = Thread(target=self._inference_worker, daemon=True)
        self._worker_thread.start()
        logger.info("Pipeline inference worker started")

    def start(self):
        self.start_background_loop()

    async def stop(self):
        if not self._running:
            try:
                self._ready_event.clear()
            except Exception:
                pass
            return

        # Request worker shutdown and wait for acknowledgement via queue
        shutdown_q: "Queue[Any]" = Queue(maxsize=1)
        try:
            self._task_queue.put(("shutdown", ("shutdown", None), shutdown_q))
        except Exception:
            logger.exception("Failed to enqueue shutdown task")
            # proceed to best-effort shutdown below

        # Wait for worker to put result into shutdown_q (blocking in executor so we don't block event loop)
        try:
            loop = None
            try:
                loop = asyncio.get_event_loop()
                # run blocking get() in default executor
                await loop.run_in_executor(None, shutdown_q.get, True, 10.0)
            except RuntimeError:
                # No running event loop in this thread: create a temporary loop to wait synchronously
                temp_loop = asyncio.new_event_loop()
                try:
                    await temp_loop.run_in_executor(None, shutdown_q.get, True, 10.0)
                finally:
                    try:
                        temp_loop.close()
                    except Exception:
                        pass
        except Exception:
            # ignore waiting errors; still proceed to shutdown
            logger.debug("Timed out or error waiting for worker shutdown acknowledgement")

        # Signal worker thread to stop if not already
        try:
            self._stop_event.set()
        except Exception:
            pass

        # Join worker thread (give it some time)
        if self._worker_thread:
            try:
                self._worker_thread.join(timeout=5.0)
            except Exception:
                pass

        # Shutdown IO executor
        try:
            self._io_executor.shutdown(wait=False)
        except Exception:
            pass

        self._running = False
        try:
            self._ready_event.clear()
        except Exception:
            pass

        logger.info("Pipeline stopped")

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        try:
            return self._ready_event.wait(timeout=timeout)
        except Exception:
            return False

    def _cleanup_gpu_resources(self):
        try:
            if getattr(self, 'crnn_model', None) is not None:
                try:
                    del self.crnn_model
                except Exception:
                    self.crnn_model = None
            if getattr(self, 'yolo_model', None) is not None:
                try:
                    del self.yolo_model
                except Exception:
                    self.yolo_model = None
            gc.collect()
        except Exception as e:
            logger.exception("GPU cleanup error: %s", e)

    def _inference_worker(self):
        try:
            import pycuda.driver as cuda
            cuda.init()
            dev = cuda.Device(0)
            self._cuda_ctx = dev.make_context()
            logger.info("CUDA context created in worker")
        except Exception as e:
            logger.warning("CUDA context not created: %s", e)
            self._cuda_ctx = None

        try:
            self.ensure_output_dirs()
            self.load_models()
        except Exception as e:
            logger.exception("Worker init error: %s", e)

        while not self._stop_event.is_set():
            try:
                task_id, payload, result_q = self._task_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                kind, data = payload
                if kind == "bytes":
                    arr = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        res = {"error": "cannot decode frame"}
                    else:
                        res = self._process_frame_sync(frame)
                elif kind == "np":
                    res = self._process_frame_sync(data)
                elif kind == "shutdown":
                    try:
                        self._cleanup_gpu_resources()
                    except Exception:
                        pass
                    res = {"status": "shutdown_ok"}
                    try:
                        result_q.put(res, block=False)
                    except Exception:
                        pass
                    self._task_queue.task_done()
                    break
                else:
                    res = {"error": "unknown payload kind"}
            except Exception as e:
                logger.exception("Worker exception: %s", e)
                res = {"error": f"worker exception: {e}"}

            try:
                result_q.put(res, block=False)
            except Exception:
                pass
            self._task_queue.task_done()

        try:
            self._cleanup_gpu_resources()
        except Exception:
            pass

        try:
            if self._cuda_ctx is not None:
                try:
                    self._cuda_ctx.pop()
                except Exception:
                    pass
                try:
                    self._cuda_ctx.detach()
                except Exception:
                    pass
                self._cuda_ctx = None
                logger.info("CUDA context popped in worker")
        except Exception as e:
            logger.exception("CUDA cleanup failed: %s", e)

        logger.info("Inference worker exiting")

    async def _broadcast(self, message: dict):
        dead = []
        for ws in list(self._ws_clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for d in dead:
            try:
                self.unregister_ws(d)
            except Exception:
                pass

    def _raw_send_to_ws(self, payload: dict):
        """
        Low-level send implementation: posts to configured webserver push endpoint.
        Kept separate so we can wrap with safe_send_webpayload (sanitize + dedupe).
        """
        try:
            webserver_url = str(self.processing_cfg.get("webserver_url", "http://127.0.0.1:8080")).rstrip("/")
            push_url = webserver_url + "/api/push"
            # Use a short timeout; failure is non-fatal
            requests.post(push_url, json=payload, timeout=1.0)
        except Exception as e:
            logger.debug("web push failed: %s", e)

    def _post_to_webserver(self, payload: dict):
        try:
            # Ensure outgoing payload contains the raw OCR/checks/merged fields needed by the backend.
            # Do NOT include server-side filter metadata here; backend will be responsible for aggregation.
            logger.debug("[PIPELINE->WEB] Posting to webserver (merged=%s) webserver_url=%s",
                        payload.get("merged"),
                        str(self.processing_cfg.get("webserver_url", "http://127.0.0.1:8080")))
            safe_send_webpayload(self, payload, lambda p: self._raw_send_to_ws(p))
            logger.debug("[PIPELINE->WEB] POST done for uid=%s", payload.get("uid"))
        except Exception:
            logger.exception("Failed to _post_to_webserver")

    # Replace the existing _add_recent_result with this version
    def _add_recent_result(self, entry: dict, class1_check: dict, class2_check: dict, class1_merged: str, class2_merged: str):
        try:
            now = time.time()
            rec = {
                "ts": now,
                "uid": entry.get("uid"),
                # store image and detections so emit handler can fall back to recent data
                "image_b64": entry.get("image_b64", ""),
                "detections": entry.get("detections", []) or [],
                "class1_merged": class1_merged or "",
                "class2_merged": class2_merged or "",
                "class1_check": class1_check or {},
                "class2_check": class2_check or {},
            }
            with self._recent_lock:
                self._recent_results.append(rec)
                cutoff = now - (max(5.0, self.filter_interval * 10))
                self._recent_results = [r for r in self._recent_results if r["ts"] >= cutoff]
        except Exception:
            logger.exception("Failed in _add_recent_result debugging log")

    # ----------------------------
    # Public async ingress points with queue-backpressure
    # ----------------------------
    async def process_bytes(self, data: bytes):
        try:
            qsize = self._task_queue.qsize()
        except Exception:
            qsize = 0
        if qsize >= self.max_task_queue and self.drop_frames_when_busy:
            return {"dropped": True, "reason": "queue_full"}
        result_q: "Queue[Any]" = Queue(maxsize=1)
        task_id = str(uuid.uuid4())[:8]
        self._task_queue.put((task_id, ("bytes", data), result_q))
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, result_q.get)
        return res

    async def process_frame(self, frame_bgr: np.ndarray):
        try:
            qsize = self._task_queue.qsize()
        except Exception:
            qsize = 0
        if qsize >= self.max_task_queue and self.drop_frames_when_busy:
            return {"dropped": True, "reason": "queue_full"}
        result_q: "Queue[Any]" = Queue(maxsize=1)
        task_id = str(uuid.uuid4())[:8]
        self._task_queue.put((task_id, ("np", frame_bgr), result_q))
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, result_q.get)
        return res

    def _crnn_infer_with_inflight(self, pil_img: Image.Image):
        """
        Helper to wrap CRNN inference with _ocr_inflight tracking.
        Returns the inference result dict from models.crnn_infer.
        """
        with self._ocr_lock:
            self._ocr_inflight += 1
        try:
            res = models.crnn_infer(self.crnn_model, pil_img)
            return res
        finally:
            with self._ocr_lock:
                try:
                    self._ocr_inflight = max(0, self._ocr_inflight - 1)
                except Exception:
                    self._ocr_inflight = max(0, getattr(self, "_ocr_inflight", 0) - 1)
                try:
                    self._ocr_cond.notify_all()
                except Exception:
                    pass

    def _process_frame_sync(self, frame_bgr: np.ndarray):
        # safe frame_id generation (thread-safe)
        try:
            with self._frame_seq_lock:
                self._frame_seq += 1
                frame_id = self._frame_seq
        except Exception:
            self._frame_seq = getattr(self, "_frame_seq", 0) + 1
            frame_id = self._frame_seq

        ts_float = time.time()
        ts = int(ts_float)
        uid = str(uuid.uuid4())[:8]
        display_time = make_display_timestamp(ts_float)

        jpeg_q = self.stream_jpeg_quality

        # ensure dims early (fix ordering bug)
        H_img, W_img = frame_bgr.shape[:2]

        # Update runtime orig_size so set_roi and others see current frame size
        try:
            # authoritative runtime size
            self.orig_size = [int(W_img), int(H_img)]
            # also keep last_frame_size tuple for compatibility with other code paths
            self.last_frame_size = (int(W_img), int(H_img))
        except Exception:
            pass

        # --- NEW: apply pending normalized ROI on first frame if present ---
        try:
            if getattr(self, "_pending_roi", None) and not getattr(self, "_roi_applied_once", False):
                pending = getattr(self, "_pending_roi", None)
                try:
                    if pending.get("normalized"):
                        cam_w, cam_h = self.orig_size[0], self.orig_size[1]
                        xi = int(round(pending["x"] * cam_w))
                        yi = int(round(pending["y"] * cam_h))
                        wi = int(round(pending["w"] * cam_w))
                        hi = int(round(pending["h"] * cam_h))
                    else:
                        xi = int(pending.get("x", 0)); yi = int(pending.get("y", 0))
                        wi = int(pending.get("w", 0)); hi = int(pending.get("h", 0))
                    self.set_roi(xi, yi, wi, hi)
                    logger.info("Applied pending ROI on first frame: %s", getattr(self, "_roi", None))
                except Exception:
                    logger.exception("Failed to apply pending ROI on first frame")
                try:
                    self._pending_roi = None
                except Exception:
                    self._pending_roi = None
                self._roi_applied_once = True
        except Exception:
            logger.exception("Pending ROI application check failed")

        # --- NEW: runtime resolution change detection (debounced) and auto re-apply saved norm ROI ---
        try:
            prev = getattr(self, "_prev_frame_size", None)
            candidate = getattr(self, "_last_size_candidate", None)
            new_size = self.last_frame_size
            if prev != new_size:
                if candidate == new_size:
                    self._size_stable_count = (getattr(self, "_size_stable_count", 0) or 0) + 1
                else:
                    self._last_size_candidate = new_size
                    self._size_stable_count = 1
                logger.debug("Frame size candidate=%s (count=%d) prev=%s", new_size, self._size_stable_count, prev)
            else:
                self._last_size_candidate = None
                self._size_stable_count = 0

            if (getattr(self, "_size_stable_count", 0) >= max(1, int(self.processing_cfg.get("resolution_stable_threshold", 3)))):
                accepted = self._last_size_candidate or new_size
                old = getattr(self, "_prev_frame_size", None)
                if old != accepted:
                    try:
                        self._prev_frame_size = accepted
                        logger.info("Runtime frame size stabilized: %s (was %s)", accepted, old)
                        # 1) if pending exists, apply (already handled above)
                        # 2) else, try applying saved normalized ROI from settings_manager if present
                        sm = getattr(self, "settings_manager", None)
                        if sm is not None:
                            s = sm.all() or {}
                            p = s.get("processing", {}) or {}
                            saved_norm = p.get("roi", None)
                            if saved_norm and isinstance(saved_norm, dict):
                                try:
                                    px_x = int(round(float(saved_norm["x"]) * accepted[0]))
                                    px_y = int(round(float(saved_norm["y"]) * accepted[1]))
                                    px_w = int(round(float(saved_norm["w"]) * accepted[0]))
                                    px_h = int(round(float(saved_norm["h"]) * accepted[1]))
                                    self.set_roi(px_x, px_y, px_w, px_h)
                                    logger.info("Applied saved normalized ROI after resolution change: %s", getattr(self, "_roi", None))
                                except Exception:
                                    logger.exception("Failed to apply saved normalized ROI after resolution change")
                        # 3) if we had a stored pixel ROI and prior orig_size known, re-normalize -> reapply
                        cur = getattr(self, "_roi", None)
                        prior_orig = getattr(self, "orig_size", None)
                        # Note: prior_orig was just set to new size; if you want prior prior size, keep separate attr — skipped for simplicity
                        # If you want to rebase from previously known orig_size, store it separately when set_roi is called.
                    except Exception:
                        logger.exception("Error while handling resolution stabilization")
                self._size_stable_count = 0
                self._last_size_candidate = None
        except Exception:
            logger.exception("Resolution change handling error")

        detections = []
        annotated = None  # allocate later if needed

        # === YOLO inference + roi(with optional throttling) ===
        roi_local = None
        with self._roi_lock:
            if self._roi:
                roi_local = dict(self._roi)

        crop_frame = None
        ex1 = ey1 = 0
        if roi_local:
            rx = max(0, int(roi_local.get("x", 0)))
            ry = max(0, int(roi_local.get("y", 0)))
            rw = max(0, int(roi_local.get("w", 0)))
            rh = max(0, int(roi_local.get("h", 0)))
            if rw <= 0 or rh <= 0 or rx >= W_img or ry >= H_img:
                logger.debug("Invalid ROI, running full-frame YOLO instead")
                crop_frame = None
            else:
                ex2 = min(W_img, rx + rw)
                ey2 = min(H_img, ry + rh)
                ex1 = max(0, rx)
                ey1 = max(0, ry)
                # crop_frame uses numpy slicing [y1:y2, x1:x2]
                crop_frame = frame_bgr[ey1:ey2, ex1:ex2]

        if self.yolo_model is None:
            logger.debug("No YOLO model loaded")
            dets = []
        else:
            run_yolo = True
            # manual throttling by configured rate
            if getattr(self, "initial_yolo_rate", 0.0) and self.initial_yolo_rate > 0.0:
                now_ts = time.time()
                min_interval = 1.0 / float(self.initial_yolo_rate)
                if (now_ts - getattr(self, "_last_yolo_ts", 0.0)) < min_interval:
                    run_yolo = False
                else:
                    self._last_yolo_ts = now_ts

            # auto backpressure mode: skip YOLO if OCR in-flight
            if self.yolo_mode == "auto":
                with self._ocr_lock:
                    if self._ocr_inflight > 0:
                        run_yolo = False

            if not run_yolo:
                logger.debug("[YOLO-SKIP] skipping YOLO (throttle/auto-backpressure)")
                dets = []
            else:
                with self._time_block("YOLO"):
                    try:
                        # use configured conf threshold (fallback to 0.4)
                        conf_thres = float(self.yolo_cfg.get("conf", 0.4))
                        iou_thres = float(self.yolo_cfg.get("iou", 0.45))
                        if crop_frame is not None and crop_frame.size != 0:
                            dets_crop = models.yolo_infer(self.yolo_model, crop_frame,
                                                          input_size=self.yolo_cfg.get("imgsz", 640),
                                                          conf_thres=conf_thres,
                                                          iou_thres=iou_thres)
                            dets = []
                            for rc in dets_crop:
                                #  헬퍼 사용: ROI offset 적용 + clamp
                                clamped = clamp_bbox(rc.get("bbox", [0, 0, 0, 0]), W_img, H_img, offset=(ex1, ey1))
                                if clamped:
                                    new_rc = dict(rc)
                                    new_rc["bbox"] = clamped
                                    dets.append(new_rc)
                        else:
                            dets = models.yolo_infer(self.yolo_model, frame_bgr,
                                                     input_size=self.yolo_cfg.get("imgsz", 640),
                                                     conf_thres=conf_thres,
                                                     iou_thres=iou_thres)
                    except Exception as e:
                        logger.exception("yolo_infer failed: %s", e)
                        dets = []

        # If no detections: heartbeat / optional streaming / include_empty handling
        if not dets:
            now_ts = time.time()
            if self.heartbeat_interval > 0 and (now_ts - getattr(self, "_last_heartbeat_ts", 0.0)) >= self.heartbeat_interval:
                logger.info("[HEARTBEAT] no detections ")
                self._last_heartbeat_ts = now_ts

            should_stream = False
            # Only stream empty frames in 'always' mode (respect stream_interval)
            if self.stream_mode == "always":
                if self.stream_interval <= 0.0 or (now_ts - getattr(self, "_last_stream_ts", 0.0)) >= self.stream_interval:
                    should_stream = True
                    self._last_stream_ts = now_ts

            # If 'stop' or 'detect' modes, should_stream remains False (detect streams only when detections exist)
            if should_stream:
                try:                    
                    send_img = self._resize_for_streaming(frame_bgr)                  
                    image_b64, image_data_uri = encode_image_to_base64(send_img, jpeg_q, with_data_uri=True)            
                except Exception:
                    logger.exception("Failed to prepare empty-frame for websocket streaming")
                    image_b64 = ""
                    image_data_uri = ""
                    send_img = frame_bgr

                # ✅ 헬퍼 사용
                send_h, send_w = send_img.shape[:2] if send_img is not None else (H_img, W_img)
                clients_payload = self._build_clients_payload(
                    uid=uid,
                    ts=ts,
                    frame_id=frame_id,
                    orig_size=(W_img, H_img),
                    sent_size=(send_w, send_h),
                    detections=[],
                    merged={"class1": "", "class2": ""},
                    checks={},
                    image_b64=image_b64,
                    image_data_uri=image_data_uri,
                    roi=None
                )

                try:
                    self._add_recent_result(clients_payload, {"status": "NONE"}, {"status": "NONE"}, "", "")
                except Exception:
                    pass

                try:
                    # post to webserver without embedding large raw frame (if any)
                    post_payload = dict(clients_payload)
                    self._io_executor.submit(self._post_to_webserver, post_payload)
                except Exception:
                    try:
                        self._post_to_webserver(post_payload)
                    except Exception:
                        logger.exception("Failed to post empty-frame payload synchronously")

            # ✅ 헬퍼 사용
            if not getattr(self, "include_empty", False):
                payload_for_broadcast = self._build_clients_payload(
                    uid=uid,
                    ts=ts_float,
                    frame_id=frame_id,
                    orig_size=(W_img, H_img),
                    sent_size=(W_img, H_img),
                    detections=[],
                    merged={"class1": "", "class2": ""},
                    checks={},
                    image_b64="",
                    image_data_uri="",
                    roi=None
                )
                return payload_for_broadcast

            payload_for_broadcast = self._build_clients_payload(
                uid=uid,
                ts=ts_float,
                frame_id=frame_id,
                orig_size=(W_img, H_img),
                sent_size=(W_img, H_img),
                detections=[],
                merged={"class1": "", "class2": ""},
                checks={"class1": {"status": "NONE"}, "class2": {"status": "NONE"}},
                image_b64="",
                image_data_uri="",
                roi=None
            )
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self._broadcast(payload_for_broadcast))
            except Exception:
                logger.exception("Failed to broadcast empty-frame payload")
            return payload_for_broadcast

        # detections exist -> full processing
        try:
            sep_len = max(_MIN_SEPARATOR_LEN, min(_MAX_SEPARATOR_LEN, 40))
            sep = "-" * sep_len
            print(sep, file=sys.stderr)
        except Exception:
            pass

        try:
            annotated = frame_bgr.copy()
        except Exception:
            annotated = None

        # per-detection OCR and crop/save
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = map(int, d['bbox'])
            conf = float(d.get('conf', 0.0))
            cls = str(d.get('class', '0')).strip()

            expanded = clamp_bbox(d['bbox'], W_img, H_img, expand_pad=8)
            if not expanded:
                continue
            ex1, ey1, ex2, ey2 = expanded

            crop = np.ascontiguousarray(frame_bgr[ey1:ey2, ex1:ex2])

            ocr_text = None
            ocr_conf = 0.0

            if cls != "0" and self.crnn_model is not None and crop.size != 0:
                try:
                    pil_orig = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    pil_gray = pil_orig.convert('L')
                    pil_enh = Image.fromarray(enhance_gray_for_crnn(crop)).convert('L')

                    candidates = build_candidates(pil_orig, pil_gray, pil_enh, self.candidate_list)

                    best = None
                    best_conf = -1.0
                    best_pil_for_ocr = None

                    if self.candidate_mode == "all":
                        for cand in candidates:
                            with self._time_block("CRNN-candidate"):
                                res_c = self._crnn_infer_with_inflight(cand)
                            conf_c = float(res_c.get('confidence', 0.0))
                            if conf_c > best_conf:
                                best_conf = conf_c
                                best = res_c
                                best_pil_for_ocr = cand
                    else:
                        if candidates:
                            with self._time_block("CRNN-candidate-first"):
                                res_c = self._crnn_infer_with_inflight(candidates[0])
                            best = res_c
                            best_conf = float(res_c.get('confidence', 0.0))
                            best_pil_for_ocr = candidates[0]

                    if best_conf < 0.6:
                        inv = Image.fromarray(255 - np.array(pil_enh)).convert('L')
                        with self._time_block("CRNN-invert"):
                            res_inv = self._crnn_infer_with_inflight(inv)
                        if float(res_inv.get('confidence', 0.0)) > best_conf:
                            best = res_inv
                            best_conf = float(res_inv.get('confidence', 0.0))
                            best_pil_for_ocr = inv

                    if best is not None:
                        ocr_text = best.get('text', '') or ''
                        ocr_conf = float(best.get('confidence', 0.0))
                    else:
                        ocr_text = None
                        ocr_conf = 0.0
                except Exception:
                    logger.exception("CRNN infer error")
                    ocr_text = None
                    ocr_conf = 0.0
            else:
                ocr_text = None
                ocr_conf = 0.0

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "class": cls,
                "crop": "",
                "ocr_text": ocr_text,
                "ocr_conf": ocr_conf
            })

        try:
            logger.debug("frame ts=%s uid=%s detections_count=%s", ts, uid, len(detections))
            for di, dd in enumerate(detections):
                logger.debug("det[%s] class=%s bbox=%s ocr_text=%s ocr_conf=%s", di, dd.get('class'), dd.get('bbox'), dd.get('ocr_text'), dd.get('ocr_conf'))
        except Exception:
            pass

        if annotated is not None:
            for d in detections:
                try:
                    x1, y1, x2, y2 = d["bbox"]
                    label = f"{d['class']}:{d['conf']:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception:
                    pass

        # build merged strings and checks
        class1_merged, class2_merged = build_parts_for_classes(detections)
        class1_check = check_container_bic(class1_merged)
        class2_check = check_class2_format(class2_merged)

        # --- NORMALIZE detections -> detections_full (ensure full-frame coords and ints) ---
        try:
            # dets should already be full-frame coords (crop case we added ex1/ey1 earlier).
            # Normalize to ints and keep as canonical source for downstream work.
            detections_full = []
            for d in (dets or []):
                dd = dict(d)
                bb = dd.get("bbox") or dd.get("box") or []
                if not bb or len(bb) < 4:
                    continue
                # 헬퍼 사용: float -> int 변환 + clamp
                clamped = clamp_bbox(bb, W_img, H_img)
                if clamped:
                    dd["bbox"] = clamped
                    detections_full.append(dd)
        except Exception:
            detections_full = []

        # --- DEBUG: log ROI used and a sample of detections to detect mismatches ---
        try:
            logger.debug("FRAME ROI used=%s  detections_full_count=%d", roi_local, len(detections_full))
            if detections_full:
                # log first two boxes
                for i, sdet in enumerate(detections_full[:2]):
                    logger.debug("SAMP_DET[%d] bbox=%s class=%s ocr_text=%s", i, sdet.get("bbox"), sdet.get("class"), sdet.get("ocr_text"))
        except Exception:
            pass

        # Optional: check whether detections lie within ROI if ROI is set (diagnostic)
        try:
            if roi_local:
                rx = int(roi_local.get("x", 0)); ry = int(roi_local.get("y", 0))
                rw = int(roi_local.get("w", 0)); rh = int(roi_local.get("h", 0))
                rx2 = rx + rw; ry2 = ry + rh
                for i, sd in enumerate(detections_full):
                    bx1, by1, bx2, by2 = sd.get("bbox")
                    # If majority of box outside ROI, log a warning (helps debug ROIs changing)
                    inter_x1 = max(bx1, rx); inter_y1 = max(by1, ry)
                    inter_x2 = min(bx2, rx2); inter_y2 = min(by2, ry2)
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    box_area = max(1, (bx2 - bx1) * (by2 - by1))
                    if box_area > 0 and (inter_area / float(box_area)) < 0.5:
                        logger.warning("DET[%d] mostly outside ROI (overlap=%.2f) roi=%s bbox=%s", i, (inter_area/float(box_area)), roi_local, sd.get("bbox"))
        except Exception:
            pass
        
        # draw annotations on full-res using detections_full (canonical source)
        try:
            annotated_full = frame_bgr.copy() if frame_bgr is not None else None
            if annotated_full is not None:
                for d in detections_full:
                    try:
                        x1, y1, x2, y2 = d.get("bbox")
                        label = f"{d.get('class')}:{float(d.get('conf',0.0)):.2f}"
                        cv2.rectangle(annotated_full, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(annotated_full, label, (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    except Exception:
                        pass
        except Exception:
            annotated_full = frame_bgr.copy() if frame_bgr is not None else None


        # Retry OCR  & postprocessing (uses config attributes defined in __init__)
        try:
            class1_idxs = [idx for idx, det in enumerate(detections) if str(det.get("class")) == "1"]
            if self.debug_retry:
                try:
                    ocr_confs = [detections[i].get("ocr_conf", 0.0) for i in class1_idxs]
                    logger.debug("FORCE-RETRY CHECK class1_idxs=%s ocr_retry_scales=%s class1_check.status=%s ocr_confs=%s force_retry_enabled=%s",
                                class1_idxs, self.ocr_retry_scales, class1_check.get('status'), ocr_confs, self.force_retry_enabled)
                except Exception:
                    pass

            need_retry_class1 = bool(class1_idxs) and (class1_check.get("status") != "OK" or any(d.get("ocr_conf", 0.0) < self.min_ocr_conf for d in [detections[i] for i in class1_idxs]))
            if need_retry_class1 and getattr(self, "ocr_retry_scales", []) and self.force_retry_enabled:
                logger.info("OCR-FORCE-RETRY attempting merged retries")
                detections = attempt_merged_retry_class1(
                    detections=detections,
                    frame_bgr=frame_bgr,
                    class1_idxs=class1_idxs,
                    ocr_retry_scales=self.ocr_retry_scales,
                    retry_candidate_list=self.retry_candidate_list,
                    retry_candidate_mode=self.retry_candidate_mode,
                    crnn_model=self.crnn_model,
                    check_container_bic_fn=check_container_bic,
                    min_accept_conf=self.min_ocr_conf,
                )
                class1_merged, class2_merged = build_parts_for_classes(detections)
                class1_check = check_container_bic(class1_merged)
            else:
                logger.debug("FORCE-RETRY SKIP need_retry_class1=%s", need_retry_class1)
        except Exception:
            logger.exception("OCR-FORCE-RETRY unexpected error")

        try:
            class2_idxs = [idx for idx, det in enumerate(detections) if str(det.get("class")) == "2"]
            need_retry_class2 = bool(class2_idxs) and (class2_check.get("status") != "OK")
            if need_retry_class2 and getattr(self, "ocr_retry_scales", []) and self.force_retry_enabled:
                logger.info("OCR-RETRY-CLASS2 attempting per-detection retries")
                detections = attempt_retry_class2(
                    detections=detections,
                    frame_bgr=frame_bgr,
                    class2_idxs=class2_idxs,
                    ocr_retry_scales=self.ocr_retry_scales,
                    retry_candidate_list=self.retry_candidate_list,
                    retry_candidate_mode=self.retry_candidate_mode,
                    crnn_model=self.crnn_model,
                    check_class2_format_fn=check_class2_format,
                    min_accept_conf=self.min_ocr_conf,
                )
                class1_merged, class2_merged = build_parts_for_classes(detections)
                class2_check = check_class2_format(class2_merged)
            else:
                logger.debug("CLASS2-RETRY SKIP need_retry_class2=%s", need_retry_class2)
        except Exception:
            logger.exception("OCR-RETRY-CLASS2 unexpected error")

        # compute checksum (cal_cd) and attempt to correct status if possible
        try:
            computed_cal_cd = None
            if class1_merged:
                s = class1_merged.strip().upper()
                m = getattr(models, "_CONTAINER_RE", None) or None
                # fallback to local regex if models did not expose it:
                if not m:
                    # from app.utils import _CONTAINER_RE if needed, but here we rely on check_container_bic's regex
                    pass
                # use the same logic as check_container_bic to extract parts
                try:
                    # reuse _CONTAINER_RE via check logic: parse s manually
                    from app.utils import _CONTAINER_RE as __CR
                    m2 = __CR.match(s)
                except Exception:
                    m2 = None
                if m2:
                    owner_part = m2.group(1)
                    digits7 = m2.group(2)
                    serial6 = digits7[:6]
                    try:
                        computed_cal_cd = calculate_checksum_custom(owner_part, serial6)
                    except Exception:
                        computed_cal_cd = None
                    if computed_cal_cd is not None and computed_cal_cd != "?" and class1_check.get('ocr_cd') is None:
                        try:
                            class1_check['ocr_cd'] = int(digits7[6])
                        except Exception:
                            pass
                    if computed_cal_cd is not None and computed_cal_cd != "?" and class1_check.get('ocr_cd') is not None:
                        try:
                            if str(class1_check['ocr_cd']).strip() == str(computed_cal_cd).strip():
                                prev = class1_check.get('status')
                                class1_check['cal_cd'] = computed_cal_cd
                                class1_check['status'] = 'OK'
                                logger.info("CALCD-CORRECT class1 status corrected %s -> OK for '%s' (ocr_cd=%s cal_cd=%s)",
                                            prev, class1_merged, class1_check.get('ocr_cd'), computed_cal_cd)
                        except Exception:
                            pass
        except Exception:
            logger.exception("cal_cd computation error")

        # prepare send_img (resized annotated) so annotations scale correctly on clients
        try:          
            base_img = annotated_full if annotated_full is not None else frame_bgr
            send_img = self._resize_for_streaming(base_img)
            send_h, send_w = send_img.shape[:2] if send_img is not None else (H_img, W_img)
        except Exception:
            base_img = annotated_full if annotated_full is not None else frame_bgr
            send_img = base_img
            send_h, send_w = (send_img.shape[0], send_img.shape[1]) if send_img is not None else (H_img, W_img)
        # scaled detections for UI overlay (헬퍼 사용)
        scaled_detections = scale_detections(detections_full, (W_img, H_img), (send_w, send_h))

        # --- Encode annotated image for clients (BACKUP behavior restored) ---

        image_b64 = ""
        image_data_uri = ""
        try:
            # decide send policy:
            should_send_annotated = False
            if self.stream_mode == "always":
                should_send_annotated = True
            elif self.stream_mode == "detect":
                should_send_annotated = bool(detections_full)
            elif self.stream_mode == "stop":
                should_send_annotated = bool(detections_full)
            else:
                should_send_annotated = bool(detections_full)

            if should_send_annotated and send_img is not None:
                image_b64, image_data_uri = encode_image_to_base64(send_img, jpeg_q, with_data_uri=True)
                # cache for stop mode
                try:
                    self._last_annotated_image_b64 = image_b64
                    self._last_annotated_image_data_uri = image_data_uri
                    self._last_annotated_timestamp = ts_float
                except Exception:
                    pass
            else:
                if getattr(self, "stream_mode", "detect") == "stop":
                    image_b64 = getattr(self, "_last_annotated_image_b64", "") or ""
                    image_data_uri = getattr(self, "_last_annotated_image_data_uri", "") or ""
        except Exception:
            logger.exception("Failed to encode annotated image for clients")
            image_b64 = ""
            image_data_uri = ""

        # --- Encode raw frame for internal storage only (do not broadcast raw frame to clients) ---
        frame_b64, _ = encode_image_to_base64(frame_bgr, jpeg_q, with_data_uri=False)

 
        # --- Build clients payload (ANNOTATED) ---
        # ✅ 헬퍼 사용
        clients_payload = self._build_clients_payload(
            uid=uid,
            ts=ts_float,
            frame_id=frame_id,
            orig_size=(W_img, H_img),
            sent_size=(send_w, send_h),
            detections=scaled_detections,
            merged={"class1": class1_merged, "class2": class2_merged},
            checks={"class1": class1_check, "class2": class2_check},
            image_b64=image_b64 or "",
            image_data_uri=image_data_uri or "",
            roi=roi_local
        )

        # --- Build internal payload ---
        # ✅ 헬퍼 사용
        internal_payload = self._build_internal_payload(
            uid=uid,
            ts=ts_float,
            frame_id=frame_id,
            orig_size=(W_img, H_img),
            sent_size=(send_w, send_h),
            detections_full=detections_full,
            merged={"class1": class1_merged, "class2": class2_merged},
            checks={"class1": class1_check, "class2": class2_check},
            frame_b64=frame_b64,
            roi=roi_local
        )

        # Debug log to confirm what we broadcast
        # --- Create a small union-crop from class1/class2 detections and cache it ---

        try:
            # ✅ 헬퍼 메서드 사용
            union_bbox = compute_union_bbox(detections_full)
            if union_bbox:
                # ✅ 헬퍼 메서드가 패딩까지 처리하므로 원본 bbox 그대로 전달
                crop_bytes = create_crop_bytes(
                    frame_bgr, union_bbox, pad=self.union_crop_pad, target_width=self.union_crop_target_width, jpeg_quality=self.union_crop_jpeg_quality
                )
                if crop_bytes:
                    try:
                        from app.utils import crop_cache
                        crop_id = crop_cache.put(crop_bytes)
                        
                        # 패딩 적용된 최종 bbox 계산 (로깅용)
                        pad = self.union_crop_pad
                        ux1, uy1, ux2, uy2 = union_bbox
                        padded_bbox = [
                            max(0, ux1 - pad),
                            max(0, uy1 - pad),
                            min(W_img, ux2 + pad),
                            min(H_img, uy2 + pad)
                        ]
                        
                        internal_payload["_union_crop_ref"] = {
                            "type": "mem",
                            "id": crop_id,
                            "len": len(crop_bytes),
                            "bbox": padded_bbox,
                            "ts": ts_float,
                        }
                        logger.debug("pipeline: union crop created id=%s len=%d uid=%s", crop_id, len(crop_bytes), uid)
                    except Exception:
                        logger.exception("Failed to put union crop into cache")
        except Exception:
            logger.exception("Union crop overall exception")
        
        try:
            logger.info("BROADCAST->WEB DEBUG uid=%s mode=%s annotated_len=%d frame_len=%d dets=%d",
                        uid, self.stream_mode, len(clients_payload.get("image_b64") or ""), len(internal_payload.get("frame_b64") or ""), len(detections_full))
        except Exception:
            pass

        # 1) Send clients_payload to webserver so webserver will broadcast the annotated image to WS clients.
        #    This centralizes broadcasting and avoids calling asyncio loop from worker threads.
        try:
            # POST clients_payload (so webserver's broadcast receives annotated image)
            post_clients = dict(clients_payload)
            # Optionally, include a marker so safe_send_webpayload doesn't strip annotated image unintentionally:
            post_clients["_force_image_b64"] = True
            self._io_executor.submit(self._post_to_webserver, post_clients)
        except Exception:
            try:
                # fallback sync
                post_clients = dict(clients_payload)
                post_clients["_force_image_b64"] = True
                self._post_to_webserver(post_clients)
            except Exception:
                logger.exception("Failed to post clients_payload to webserver synchronously")

        # 2) Post internal (sanitized) payload to backend or webserver for logging, but do not include heavy frame_b64.
        try:
            post_internal = dict(internal_payload)
            post_internal.pop("frame_b64", None)
            self._io_executor.submit(self._post_to_webserver, post_internal)
        except Exception:
            try:
                self._post_to_webserver(post_internal)
            except Exception:
                logger.exception("Failed to post internal_payload synchronously")

        # 3) Always feed internal payload to FilterManager/ResultManager for cropping/storage
        try:
            if getattr(self, "filter_manager", None) is not None:
                self.filter_manager.add_event(internal_payload)
        except Exception:
            logger.exception("Failed to emit payload to FilterManager (no write performed)")


        # --- Return result for caller (compatibility) ---
        return clients_payload