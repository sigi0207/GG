#!/usr/bin/env python3
# app/result_manager.py
"""
ResultManager - daily result file writer + crop saving + upstream socket & web client payload creation/sending.

Responsibilities:
- Append human-readable summary lines to daily files YYYYMMDD.txt in results_dir.
- Keep only `retention_days` most recent daily files.
- Optionally save a single union crop image per emitted result according to configurable policy.
- Build socket/web payloads and send them asynchronously.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import re
from app.utils import safe_filename  # build_parts_for_classes removed earlier
import threading
import time
import json
import base64
import socket
import logging
from typing import Optional, Dict, Any, Callable, Tuple
from app.settings_manager import SettingsManager

# image handling (optional)
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

from concurrent.futures import ThreadPoolExecutor
import shutil

try:
    import requests
except Exception:
    requests = None  # optional; handled gracefully

logger = logging.getLogger(__name__)
_CONTAINER_RE = re.compile(r"^([A-Z]{3}U)(\d{7})$")


class ResultManager:
    def __init__(
        self,
        results_dir: Optional[Path] = None,
        crops_dir: Optional[Path] = None,
        retention_days: int = 10,
        save_crop_policy: Optional[Dict[str, list]] = None,
        socket_config: Optional[Dict[str, Any]] = None,
        web_config: Optional[Dict[str, Any]] = None,
        send_socket_fn: Optional[Callable[[dict], None]] = None,
        send_web_fn: Optional[Callable[[dict], None]] = None,
        enable_per_result_json: bool = False,
        enable_per_result_annotated: bool = False,
        crops_executor_workers: int = 2,
        server_id: Optional[str] = None,  # <-- optional
    ):
        """
        Cleaner and defensive initialization for ResultManager.
        """
        # Determine a project-root-ish base (same logic used elsewhere in app).
        try:
            root = Path(__file__).resolve().parents[1]
        except Exception:
            root = Path.cwd()

        base_out = root / "out"

        # ---------- Results / Crops / Preprocessed directories ----------
        try:
            # If caller provided a path, make it absolute/resolved; otherwise use sane defaults.
            if results_dir:
                self.results_dir = Path(results_dir).resolve()
            else:
                self.results_dir = (base_out / "results").resolve()
            self.results_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # fallback to local ./out/results
            try:
                self.results_dir = (Path.cwd() / "out" / "results").resolve()
                self.results_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.exception("Failed to initialize results_dir; using cwd")
                self.results_dir = Path.cwd()

        try:
            if crops_dir:
                self.crops_dir = Path(crops_dir).resolve()
            else:
                # prefer base_out/crops (keeps historical layout)
                self.crops_dir = (base_out / "crops").resolve()
            self.crops_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            try:
                self.crops_dir = (self.results_dir / "crops").resolve()
                self.crops_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.exception("Failed to initialize crops_dir; using results_dir/crops fallback")
                self.crops_dir = self.results_dir / "crops"

        try:
            # preprocessed dir: either provided via web_config or default to base_out/preprocessed
            self.preprocessed_dir = (base_out / "preprocessed").resolve()
            self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            try:
                self.preprocessed_dir = (self.results_dir / "preprocessed").resolve()
                self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.exception("Failed to initialize preprocessed_dir; using results_dir")
                self.preprocessed_dir = self.results_dir

        # ---------- retention / locks / policies ----------
        try:
            self.retention_days = max(1, int(retention_days or 10))
        except Exception:
            self.retention_days = 10

        self._lock = threading.Lock()
        self.save_crop_policy = save_crop_policy or {}
        self._last_cleanup_date = None

        # ---------- Socket / Web configuration normalization ----------
        # Ensure dictionaries (avoid None)
        self.socket_config = socket_config if isinstance(socket_config, dict) else (socket_config or {})
        self.web_config = web_config if isinstance(web_config, dict) else (web_config or {})

        self.send_socket_fn = send_socket_fn
        self.send_web_fn = send_web_fn

        # Normalize socket config
        if self.socket_config:
            try:
                self.socket_host = str(self.socket_config.get("host", "127.0.0.1"))
                self.socket_port = int(self.socket_config.get("port", 0) or 0)
            except Exception:
                self.socket_host = None
                self.socket_port = None
        else:
            self.socket_host = None
            self.socket_port = None
        try:
            self.socket_timeout = float(self.socket_config.get("timeout", 1.0) or 1.0)
        except Exception:
            self.socket_timeout = 1.0

        # Normalize web config
        try:
            self.webserver_url = self.web_config.get("webserver_url") or None
        except Exception:
            self.webserver_url = None
        try:
            self.web_timeout = float(self.web_config.get("timeout") or 1.0)
        except Exception:
            self.web_timeout = 1.0

        # ---------- frame_store_dir disabled ----------
        # Frame archival (saving raw frames under results/frames/...) is intentionally disabled.
        # Keep web_config for other uses, but do not create a frame_store_dir nor save frames to disk.
        self.frame_store_dir = None
        self.frame_retention_days = 0

        # web-configured thresholds
        try:
            self.min_crop_area_pixels = int(self.web_config.get("min_crop_area_pixels") or 1000)
        except Exception:
            self.min_crop_area_pixels = 1000

        # ftp upload config (defensive)
        try:
            self.ftp_config = self.web_config.get("ftp_upload", {}) or {}
        except Exception:
            self.ftp_config = {}

        # ---------- executors for async IO ----------
        try:
            workers = max(1, int(crops_executor_workers or 2))
        except Exception:
            workers = 2
        self._io_executor = ThreadPoolExecutor(max_workers=workers)
        self._ftp_executor = ThreadPoolExecutor(max_workers=1)

        # ---------- feature flags ----------
        self.enable_per_result_json = bool(enable_per_result_json)
        self.enable_per_result_annotated = bool(enable_per_result_annotated)

        # ---------- final sanity log ----------
        logger.info(
            "ResultManager initialized: results_dir=%s crops_dir=%s preprocessed_dir=%s frame_store_dir=%s retention_days=%d",
            str(self.results_dir), str(self.crops_dir), str(self.preprocessed_dir), str(self.frame_store_dir), int(self.retention_days)
        )

        # server id handling
        self.server_id = self._validate_server_id(server_id)
        logger.info("ResultManager server_id=%s", self.server_id)



    def attach_settings_manager(self, settings_manager: SettingsManager):
        try:
            sm = settings_manager
            if sm is None:
                return False
            # idempotent attach
            if getattr(self, "_attached_sm", None) is sm:
                return True
            self._attached_sm = sm

            # apply initial Server.id if available
            sid = None
            try:
                if hasattr(sm, "get_server_id"):
                    sid = sm.get_server_id(default=None)
            except Exception:
                sid = None
            if not sid:
                try:
                    s = sm.all().get("Server")
                    if isinstance(s, dict):
                        sid = s.get("id")
                    elif isinstance(s, str):
                        import re
                        m = re.search(r"id\s*:\s*([A-Za-z0-9]{1,10})", s, re.IGNORECASE)
                        if m:
                            sid = m.group(1)
                except Exception:
                    sid = None

            if sid:
                try:
                    newid = self._validate_server_id(sid)
                    if newid != "XXX" or getattr(self, "server_id", None) == "XXX":
                        self.server_id = newid
                        logger.info("ResultManager.server_id set to %s from attached SettingsManager", self.server_id)
                except Exception:
                    logger.exception("attach_settings_manager: validation failed for sid=%r", sid)

            return True
        except Exception:
            logger.exception("attach_settings_manager failed")
            return False


    ## 파이프라인에서 정보를 받지 않고, config 값을 받아 사용 ##
    def reconfigure(self, processing_cfg: Optional[dict] = None, web_config: Optional[dict] = None):
        try:
            proc = processing_cfg or {}
            web = web_config if web_config is not None else (proc.get("web") or {})

            # Make reconfigure thread-safe
            with self._lock:
                # save_crop_policy
                scp = proc.get("save_crop_policy")
                if scp is not None and isinstance(scp, dict):
                    self.save_crop_policy = scp

                # retention_days
                if proc.get("retention_days") is not None:
                    try:
                        self.retention_days = max(1, int(proc.get("retention_days")))
                    except Exception:
                        pass

                # results / crops / preprocessed dirs (truthy values override; otherwise keep existing)
                try:
                    rd = proc.get("output_dirs", {}).get("results") or proc.get("results_dir")
                    if rd:
                        self.results_dir = Path(rd).resolve()
                        self.results_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    logger.exception("ResultManager.reconfigure: failed to set results_dir")

                try:
                    cd = proc.get("output_dirs", {}).get("crops") or proc.get("crops_dir")
                    if cd:
                        self.crops_dir = Path(cd).resolve()
                        self.crops_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    logger.exception("ResultManager.reconfigure: failed to set crops_dir")

                try:
                    pd = proc.get("output_dirs", {}).get("preprocessed") or proc.get("preprocessed_dir")
                    if pd:
                        self.preprocessed_dir = Path(pd).resolve()
                        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    logger.exception("ResultManager.reconfigure: failed to set preprocessed_dir")

                try:
                    mcap = web.get("min_crop_area_pixels")
                    if mcap is not None:
                        self.min_crop_area_pixels = int(mcap)
                except Exception:
                    pass

                # webserver/url/timeout
                try:
                    if web.get("webserver_url") is not None:
                        self.webserver_url = web.get("webserver_url")
                    if web.get("timeout") is not None:
                        self.web_timeout = float(web.get("timeout"))
                except Exception:
                    pass

                # ftp config
                try:
                    ftp = web.get("ftp_upload")
                    if ftp is not None:
                        self.ftp_config = ftp or {}
                except Exception:
                    pass

            logger.info("ResultManager.reconfigure applied: results=%s crops=%s frames=%s retention=%s",
                        getattr(self, "results_dir", None), getattr(self, "crops_dir", None),
                        getattr(self, "frame_store_dir", None), getattr(self, "retention_days", None))
        except Exception:
            logger.exception("ResultManager.reconfigure failed")

    def _validate_server_id(self, sid: Optional[str]) -> str:
        try:
            if not sid:
                return "XXX"
            s = str(sid).strip().upper()
            if len(s) == 3 and s.isalnum():
                return s
        except Exception:
            pass
        logger.warning("Invalid server_id provided (%r); using fallback 'XXX'", sid)
        return "XXX"

    def _build_result_basename(self, payload: dict) -> str:
        # timestamp -> datepart/timestr
        ts = None
        for k in ("timestamp", "ts", "time"):
            if payload.get(k) is not None:
                try:
                    ts = float(payload.get(k)); break
                except Exception:
                    pass
        if ts is None:
            ts = time.time()
        dt = datetime.fromtimestamp(ts)
        datepart = dt.strftime("%Y%m%d")
        timestr = dt.strftime("%H%M%S")

        srv = getattr(self, "server_id", "XXX") or "XXX"

        merged = payload.get("merged") or {}
        class1_raw = str(merged.get("class1") or "").strip().upper()
        class2 = str(merged.get("class2") or "").strip()

        checks = payload.get("checks") or {}
        c1_check = checks.get("class1") or {}
        status = (c1_check.get("status") or "").upper()

        # INVALID policy: if status missing or 'INVALID' -> write INVAL form
        if not status or status == "INVALID":
            safe_c1 = safe_filename(class1_raw) or "UNKNOWN"
            safe_c2 = safe_filename(class2) or ""
            return f"{datepart}_{timestr}_{srv}_{safe_c1}_INVAL_{safe_c2}"

        # expect 4 letters + 7 digits
        m = re.match(r"^([A-Z]{4})(\d{7})$", class1_raw)
        if not m:
            safe_c1 = safe_filename(class1_raw) or "UNKNOWN"
            safe_c2 = safe_filename(class2) or ""
            return f"{datepart}_{timestr}_{srv}_{safe_c1}_INVAL_{safe_c2}"

        owner = m.group(1)
        digits7 = m.group(2)
        serial6 = digits7[:6]
        ocr_cd_from_digits = digits7[6]

        cal_cd = c1_check.get("cal_cd")
        try:
            if cal_cd is None:
                raise ValueError("missing cal_cd")
            cal_cd_str = str(int(cal_cd))
        except Exception:
            safe_c1 = safe_filename(class1_raw) or "UNKNOWN"
            safe_c2 = safe_filename(class2) or ""
            return f"{datepart}_{timestr}_{srv}_{safe_c1}_INVAL_{safe_c2}"

        safe_class2 = safe_filename(class2) or ""
        owner_serial = f"{owner}{serial6}"
        ocr_cd = ocr_cd_from_digits

        # Final formatted basename:
        # {YYYYMMDD}_{HHMMSS}_{SRV}_{OWNER}{SERIAL6}_{OCR_CD}_({CAL_CD})_{CLASS2}
        return f"{datepart}_{timestr}_{srv}_{owner_serial}_{ocr_cd}_({cal_cd_str})_{safe_class2}"

    # -------------------------
    # File / line formatting
    # -------------------------
    def _format_line(self, payload: dict) -> str:
        try:
            return self._build_result_basename(payload)
        except Exception:
            logger.exception("Failed to format line")
            ts = payload.get("timestamp") or payload.get("ts") or time.time()
            dt = datetime.fromtimestamp(float(ts))
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _daily_path_for_ts(self, ts: float) -> Path:
        dt = datetime.fromtimestamp(ts)
        fname = dt.strftime("%Y%m%d") + ".txt"
        return self.results_dir / fname

    def _cleanup_old_files_if_needed(self, now_ts: float):
        try:
            cutoff = datetime.fromtimestamp(now_ts).date().toordinal() - int(self.retention_days)
            for p in self.results_dir.iterdir():
                if not p.is_file():
                    continue
                name = p.name
                if len(name) >= 12 and name.endswith(".txt"):
                    try:
                        datepart = name[:8]
                        dt = datetime.strptime(datepart, "%Y%m%d").date()
                        if dt.toordinal() < cutoff:
                            try:
                                p.unlink()
                            except Exception:
                                pass
                    except Exception:
                        continue
        except Exception:
            logger.exception("Error cleaning up old result files")

    def _cleanup_old_crops_dirs(self, keep_days: int = 3):
        try:
            dirs = []
            for p in self.crops_dir.iterdir():
                if not p.is_dir():
                    continue
                name = p.name
                if len(name) == 8 and name.isdigit():
                    dirs.append(p)
            if not dirs:
                return
            dirs_sorted = sorted(dirs, key=lambda x: x.name)
            if len(dirs_sorted) <= keep_days:
                return
            to_remove = dirs_sorted[:len(dirs_sorted) - keep_days]
            for d in to_remove:
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
        except Exception:
            logger.exception("Error cleaning up old crops dirs")

    # -------------------------
    # Crop helpers
    # -------------------------
    def _save_image_b64(self, b64: str, out_path: Path) -> Optional[Path]:
        try:
            data = base64.b64decode(b64)
            with open(out_path, "wb") as fh:
                fh.write(data)
            return out_path
        except Exception:
            logger.exception("Failed to save image b64 to %s", out_path)
            return None

    def _crop_and_write_async(self, image_b64: str, bbox: tuple, out_path: Path):
        """
        Async worker: crop image bytes and write out_path.
        Changed to avoid creating .tmp files and to resize final image width to 200 px (preserve aspect).
        """
        if cv2 is None or np is None:
            logger.warning("cv2/numpy not available; cannot crop image")
            return None
        try:
            data = base64.b64decode(image_b64)
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.exception("Failed to decode image_b64 in crop worker")
                return None

            h, w = img.shape[:2]
            bx1, by1, bx2, by2 = bbox
            try:
                bx1_f, by1_f, bx2_f, by2_f = float(bx1), float(by1), float(bx2), float(by2)
            except Exception:
                logger.exception("Invalid bbox values (non-numeric): %s", bbox)
                return None

            if 0.0 <= bx1_f <= 1.01 and 0.0 <= by1_f <= 1.01 and 0.0 <= bx2_f <= 1.01 and 0.0 <= by2_f <= 1.01:
                x1 = int(round(bx1_f * w))
                y1 = int(round(by1_f * h))
                x2 = int(round(bx2_f * w))
                y2 = int(round(by2_f * h))
            else:
                x1 = int(round(bx1_f))
                y1 = int(round(by1_f))
                x2 = int(round(bx2_f))
                y2 = int(round(by2_f))

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                logger.debug("Crop bbox invalid after normalization: %s -> (%d,%d,%d,%d)", bbox, x1, y1, x2, y2)
                return None

            crop = img[y1:y2, x1:x2]

            # Resize to width 200 px preserving aspect ratio if larger than 200
            try:
                ch, cw = crop.shape[:2]
                target_w = 200
                if cw > target_w:
                    scale = target_w / float(cw)
                    new_w = target_w
                    new_h = max(1, int(round(ch * scale)))
                    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except Exception:
                pass

            ok, buf = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                logger.exception("cv2.imencode failed for crop")
                return None

            # atomic write: write to tmp file then replace
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as fh:
                    fh.write(buf.tobytes())
                logger.info("Crop written to %s", out_path)
                return out_path
            except Exception:
                logger.exception("Failed to write crop file to %s", out_path)
                return None

        except Exception:
            logger.exception("Crop worker exception")
            return None

    # -------------------------
    # Crop saving (union bbox)
    # -------------------------

    def _save_crops_if_needed(self, payload: dict) -> Optional[Dict[str, str]]:

        """
        Simplified behavior: only consume a precomputed union crop from the in-memory cache.
        If the payload contains _chosen_union_crop_ref or _union_crop_ref (type=mem),
        pop() the bytes from the crop_cache and write a single union image named by the
        standard result basename. Do not perform frame_b64 decoding or per-class crops here.
        """
        try:
            if not self.save_crop_policy:
                return None

            # timestamp / target dir
            ts = None
            for k in ("timestamp", "ts", "time"):
                if payload.get(k) is not None:
                    try:
                        ts = float(payload.get(k))
                        break
                    except Exception:
                        pass
            if ts is None:
                ts = time.time()
            dt = datetime.fromtimestamp(ts)
            date_folder = dt.strftime("%Y%m%d")
            timestr = dt.strftime("%Y%m%d%H%M%S")
            target_dir = self.crops_dir / date_folder
            target_dir.mkdir(parents=True, exist_ok=True)

            # Only handle cached union crop refs. Do not try to recrop from frame.
            crop_ref = payload.get("_chosen_union_crop_ref") or payload.get("_union_crop_ref")
            if not (isinstance(crop_ref, dict) and crop_ref.get("type") == "mem" and crop_ref.get("id")):
                return None

            crop_id = crop_ref.get("id")
            try:
                from app.utils import crop_cache
                crop_bytes = crop_cache.pop(crop_id)
            except Exception:
                logger.exception("ResultManager: failed to pop crop from cache id=%s", crop_id)
                crop_bytes = None

            if not crop_bytes:
                # nothing to save
                logger.debug("ResultManager: cached crop id=%s not available; skipping save", crop_id)
                return None

            # build basename and write as single union image
            try:
                result_basename = self._build_result_basename(payload)
            except Exception:
                result_basename = timestr
            result_basename = safe_filename(result_basename) or timestr
            out_fname = f"{result_basename}.jpg"
            out_path = target_dir / out_fname
            try:
                # atomic write: write to tmp in same dir then replace
                tmp_path = out_path.with_suffix(".tmp")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp_path, "wb") as fh:
                    fh.write(crop_bytes)
                    fh.flush()
                    try:
                        import os
                        os.fsync(fh.fileno())
                    except Exception:
                        pass
                tmp_path.replace(out_path)
                logger.info("Crop (from cache) written to %s", out_path)
                return {"crop": str(out_path)}
            except Exception:
                logger.exception("ResultManager: failed to write crop bytes to %s", out_path)
                return None
        except Exception:
            logger.exception("Failed in simplified _save_crops_if_needed")
            return None


    # -------------------------
    # Frame archival and upload
    # -------------------------
    # Note: frame archival and cleanup functions have been intentionally removed.
    # FTP upload helper is preserved for future use.

    def _ftp_upload_file(self, local_path: str):
        try:
            cfg = self.ftp_config
            host = cfg.get("host")
            if not host:
                return
            port = int(cfg.get("port", 21) or 21)
            user = cfg.get("user")
            password = cfg.get("password")
            remote_dir = cfg.get("remote_dir", "/")
            passive = bool(cfg.get("passive", True))

            from ftplib import FTP
            with FTP() as ftp:
                ftp.connect(host, port, timeout=10)
                ftp.login(user=user, passwd=password)
                ftp.set_pasv(passive)
                try:
                    ftp.cwd(remote_dir)
                except Exception:
                    parts = [p for p in remote_dir.split("/") if p]
                    cur = ""
                    for part in parts:
                        cur = cur + "/" + part
                        try:
                            ftp.mkd(cur)
                        except Exception:
                            pass
                    try:
                        ftp.cwd(remote_dir)
                    except Exception:
                        pass
                with open(local_path, "rb") as fh:
                    ftp.storbinary(f"STOR {Path(local_path).name}", fh)
        except Exception:
            logger.exception("FTP upload failed for %s", local_path)

    # -------------------------
    # Socket / Web helpers
    # -------------------------
    def _build_socket_payload(self, payload: dict) -> dict:
        merged = payload.get("merged") or {}
        checks = payload.get("checks") or {}
        ts = int(payload.get("timestamp") or payload.get("ts") or time.time())
        uid = payload.get("uid") or "uid"
        class1 = merged.get("class1") or ""
        class2 = merged.get("class2") or ""
        c1_status = (checks.get("class1") or {}).get("status") or ""
        c2_status = (checks.get("class2") or {}).get("status") or ""
        include_image = bool(self.socket_config.get("include_image", False))
        img = payload.get("image_b64") if include_image else None

        return {
            "ts": ts,
            "uid": uid,
            "class1": class1,
            "class1_status": c1_status,
            "class2": class2,
            "class2_status": c2_status,
            "image_b64": img,
        }

    def _send_socket_internal(self, payload_dict: dict):
        if not self.socket_host or not self.socket_port:
            return
        try:
            data = (json.dumps(payload_dict) + "\n").encode("utf-8")
            with socket.create_connection((self.socket_host, self.socket_port), timeout=self.socket_timeout) as s:
                s.sendall(data)
        except Exception:
            logger.exception("Internal socket send failed")

    def _async_send_socket(self, payload_dict: dict):
        try:
            if self.send_socket_fn:
                t = threading.Thread(target=lambda: self._safe_invoke(self.send_socket_fn, payload_dict), daemon=True)
                t.start()
            elif self.socket_host and self.socket_port:
                t = threading.Thread(target=self._send_socket_internal, args=(payload_dict,), daemon=True)
                t.start()
        except Exception:
            logger.exception("Failed to schedule socket send")

    def _build_web_payload(self, payload: dict) -> dict:
        merged = payload.get("merged") or {}
        checks = payload.get("checks") or {}
        ts = int(payload.get("timestamp") or payload.get("ts") or time.time())
        uid = payload.get("uid") or "uid"
        return {
            "timestamp": ts,
            "uid": uid,
            "merged": {"class1": merged.get("class1") or "", "class2": merged.get("class2") or ""},
            "checks": {"class1": checks.get("class1") or {}, "class2": checks.get("class2") or {}},
            "image_b64": payload.get("image_b64") or "",
            "filtered": payload.get("filtered", False),
            "filter_interval": payload.get("filter_interval", 0.0),
        }

    def _send_web_internal(self, payload_dict: dict):
        if not self.webserver_url:
            return
        if requests is None:
            logger.warning("requests not installed; cannot send HTTP payload")
            return
        try:
            resp = requests.post(self.webserver_url, json=payload_dict, timeout=self.web_timeout)
            if resp.status_code >= 400:
                logger.warning("Web POST returned status %s for url=%s", resp.status_code, self.webserver_url)
        except Exception:
            logger.exception("Internal web POST failed")

    def _async_send_web(self, payload_dict: dict):
        try:
            if self.send_web_fn:
                t = threading.Thread(target=lambda: self._safe_invoke(self.send_web_fn, payload_dict), daemon=True)
                t.start()
            elif self.webserver_url:
                t = threading.Thread(target=self._send_web_internal, args=(payload_dict,), daemon=True)
                t.start()
        except Exception:
            logger.exception("Failed to schedule web send")

    def _safe_invoke(self, fn: Callable[[dict], None], payload: dict):
        try:
            fn(payload)
        except Exception:
            logger.exception("User-provided sender callback raised exception")

    # -------------------------
    # Public API
    # -------------------------
    def write(self, payload: dict):
        """
        Append daily line, optionally save crops, schedule upstream/web sends.
        Frame archival is disabled; if nothing is detected and no merged values exist,
        this function will skip writing anything.
        """
        try:
            logger.debug("DEBUG RM.write uid=%s keys=%s image_b64_len=%d",
                        payload.get("uid"),
                        list(payload.keys()),
                        len(payload.get("image_b64") or ""))
        except Exception:
            pass
        with self._lock:
            try:
                logger.info("ResultManager.write: merged=%s checks=%s detections=%d image_present=%s",
                            payload.get("merged"), payload.get("checks"),
                            len(payload.get("detections") or []),
                            bool(payload.get("frame_b64") or payload.get("image_b64")))
                ts = None
                for k in ("timestamp", "ts", "time"):
                    if payload.get(k) is not None:
                        try:
                            ts = float(payload.get(k))
                            break
                        except Exception:
                            pass
                if ts is None:
                    ts = time.time()

                try:
                    self._cleanup_old_files_if_needed(ts)
                except Exception:
                    pass

                # --- SKIP WRITING IF NO DETECTIONS AND NO MERGED VALUES ---
                dets = payload.get("detections") or payload.get("detections_full") or []
                merged = payload.get("merged") or {}
                c1 = (merged.get("class1") or "").strip()
                c2 = (merged.get("class2") or "").strip()
                if (not dets) and (not c1) and (not c2):
                    logger.debug("ResultManager.write: no detections and no merged values — skipping result write for payload uid=%s", payload.get("uid"))
                    return None

                # save union crop if policy requests it (background)
                saved_images = None
                try:
                    saved_images = self._save_crops_if_needed(payload)
                except Exception:
                    logger.exception("Saving crops failed")

                # build canonical basename once (used for daily file name and for outgoing payloads)
                try:
                    result_basename = self._build_result_basename(payload)
                except Exception:
                    result_basename = None

                # append daily text line
                try:
                    line = result_basename or self._format_line(payload)
                    path = self._daily_path_for_ts(ts)
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                except Exception:
                    logger.exception("Failed to write daily result line")

                # If we saved images, ensure returned saved_images uses the daily-basename .jpg mapping
                if saved_images and result_basename:
                    # normalize: map any saved values to daily-basename + optional suffix
                    normalized = {}
                    if "crop" in saved_images:
                        normalized["crop"] = str((self.crops_dir / datetime.fromtimestamp(ts).strftime("%Y%m%d") / (safe_filename(result_basename) + ".jpg")).resolve())
                    else:
                        if "class1" in saved_images:
                            normalized["class1"] = str((self.crops_dir / datetime.fromtimestamp(ts).strftime("%Y%m%d") / (safe_filename(result_basename) + "_C1.jpg")).resolve())
                        if "class2" in saved_images:
                            normalized["class2"] = str((self.crops_dir / datetime.fromtimestamp(ts).strftime("%Y%m%d") / (safe_filename(result_basename) + "_C2.jpg")).resolve())
                    saved_images = normalized

                # Schedule socket/web sends asynchronously
                try:
                    sock_payload = self._build_socket_payload(payload)
                    if saved_images:
                        sock_payload["_saved_files"] = saved_images
                    if result_basename:
                        sock_payload["_result_basename"] = result_basename
                    sock_payload["_server_id"] = getattr(self, "server_id", "XXX")
                    self._async_send_socket(sock_payload)
                except Exception:
                    logger.exception("Scheduling socket send failed")

                try:
                    web_payload = self._build_web_payload(payload)
                    if saved_images:
                        web_payload["_saved_files"] = saved_images
                    if result_basename:
                        web_payload["_result_basename"] = result_basename
                    web_payload["_server_id"] = getattr(self, "server_id", "XXX")
                    self._async_send_web(web_payload)
                except Exception:
                    logger.exception("Scheduling web send failed")

                # Per-result JSON / annotated saving (if enabled)
                if self.enable_per_result_json or self.enable_per_result_annotated:
                    try:
                        result_obj = {
                            "timestamp": int(ts),
                            "uid": payload.get("uid"),
                            "merged": payload.get("merged"),
                            "checks": payload.get("checks"),
                            "detections": payload.get("detections"),
                        }
                        timestr = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
                        uid_safe = str(payload.get("uid") or "uid")

                        if self.enable_per_result_json:
                            fname = f"{timestr}__{uid_safe}.json"
                            json_path = self.results_dir / fname
                            with open(json_path, "w", encoding="utf-8") as jf:
                                json.dump(result_obj, jf, ensure_ascii=False, indent=2)

                        if self.enable_per_result_annotated:
                            annotated_b64 = payload.get("annotated_b64")
                            if annotated_b64:
                                ann_fname = f"{timestr}__{uid_safe}.jpg"
                                ann_path = self.crops_dir / ann_fname
                                try:
                                    data = base64.b64decode(annotated_b64)
                                    with open(ann_path, "wb") as fh:
                                        fh.write(data)
                                except Exception:
                                    logger.exception("Failed to write annotated image file")
                    except Exception:
                        logger.exception("Failed to write per-result json/annotated (if enabled)")

                return saved_images

            except Exception:
                logger.exception("Unhandled error in ResultManager.write")
                return None