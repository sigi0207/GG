# app/filter_manager.py
"""
FilterManager - aggregation and provenance-aware selection for pipeline events.

Behavior (cleaned / focused):
- Collects events via add_event().
- Every `interval` seconds groups recent events and decides canonical class1/class2 strings.
- Prefers to emit a filtered payload whose detections and frame come from the same event when possible:
  - If a single event in the window contains both class1 and class2 detections, use that event's detections AND frame.
  - Otherwise prefer detections from the events that provided the selected texts, with a fallback to the most
    recent event that contains detections for that class.
  - If detections originate from multiple event uids we choose a primary uid (majority) for the union; otherwise
    we keep per-class provenance so downstream code can decide.
- Emits filtered payloads including provenance fields:
    - frame_source_uid
    - detections_source_uid
    - detections_by_class
    - detections_source_uid_by_class
"""
from __future__ import annotations
import time
import threading
import re
import logging
from typing import List, Callable, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class FilterManager:
    def __init__(
        self,
        emit_fn: Callable[[dict], None],
        interval: float = 1.0,
        sim_threshold: float = 0.20,
        suppress_repeat_seconds: float = 30.0,
    ):
        self.emit_fn = emit_fn
        self.interval = float(interval or 1.0)
        self.sim_threshold = float(sim_threshold)
        self.suppress_repeat_seconds = float(suppress_repeat_seconds or 30.0)

        self._lock = threading.Lock()
        self._events: List[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._last_emitted_sig = None
        self._last_emit_ts = 0.0

        self._norm_re = re.compile(r'\W+')
        ###
        # representative union-crop mapping:
        # keys: tuple (class_index_str, normalized_text) e.g. ("1","TLNU4201347")
        # value: dict { "id": <crop_id>, "conf": float_confidence, "ts": timestamp }
        # this map keeps the current best (highest confidence) crop id per text signature.
        self._rep_crop_by_sig = {}
        ###


    def start(self):
        if self._thread and self._thread.is_alive():
            return
        if self.interval <= 0:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def add_event(self, payload: dict):
        # ensure ts present
        try:
            ts = float(payload.get("ts") or payload.get("timestamp") or time.time())
        except Exception:
            ts = time.time()
        ev = dict(payload)
        ev["ts"] = ts

        # normalize merged/checks into convenient keys for grouping
        merged = ev.get("merged") or {}
        if "class1" in merged and not ev.get("class1_merged"):
            ev["class1_merged"] = merged.get("class1")
        if "class2" in merged and not ev.get("class2_merged"):
            ev["class2_merged"] = merged.get("class2")

        checks = ev.get("checks") or {}
        if "class1" in checks and not ev.get("class1_check"):
            ev["class1_check"] = checks.get("class1")
        if "class2" in checks and not ev.get("class2_check"):
            ev["class2_check"] = checks.get("class2")

        # defensive copy lists to avoid mutable reference issues later
        if "detections" in ev and isinstance(ev["detections"], list):
            ev["detections"] = [dict(d) if isinstance(d, dict) else d for d in ev["detections"]]
        if "detections_full" in ev and isinstance(ev["detections_full"], list):
            ev["detections_full"] = [dict(d) if isinstance(d, dict) else d for d in ev["detections_full"]]

        with self._lock:
            self._events.append(ev)

            ###
            # Update representative crop mapping immediately (per-class) if this event carries a union crop ref.
            try:
                # only proceed if there is a cached union crop ref
                crop_ref = ev.get("_union_crop_ref") or ev.get("_chosen_union_crop_ref")
                if isinstance(crop_ref, dict) and crop_ref.get("type") == "mem" and crop_ref.get("id"):
                    crop_id = crop_ref.get("id")
                    # helper: update rep for a class if merged text exists
                    def _maybe_update_rep(class_key: str, text_key: str):
                        txt = ev.get(text_key) or ""
                        if not txt:
                            return
                        norm = self._normalize(txt)
                        if not norm:
                            return
                        # determine confidence: prefer check confidence, then ocr_conf on top-level
                        conf_cands = []
                        try:
                            ck = ev.get(f"{class_key}_check") or (ev.get("checks") or {}).get(class_key)
                            if isinstance(ck, dict):
                                v = ck.get("confidence") or ck.get("conf")
                                if v is not None:
                                    conf_cands.append(float(v))
                        except Exception:
                            pass
                        try:
                            v2 = ev.get("ocr_conf")
                            if v2 is not None:
                                conf_cands.append(float(v2))
                        except Exception:
                            pass
                        conf_val = max(conf_cands) if conf_cands else 0.0

                        key = (class_key, norm)
                        prev = self._rep_crop_by_sig.get(key)
                        if not prev or conf_val > float(prev.get("conf", 0.0)):
                            # record new representative; try to free old crop id if present and different
                            old_id = prev.get("id") if prev else None
                            self._rep_crop_by_sig[key] = {"id": crop_id, "conf": float(conf_val), "ts": ev.get("ts", time.time())}
                            if old_id and old_id != crop_id:
                                try:
                                    # attempt to free previous crop from cache to save memory (best-effort)
                                    from app.utils import crop_cache
                                    crop_cache.pop(old_id)
                                except Exception:
                                    # don't fail the whole flow on pop error
                                    logger.debug("FilterManager: failed to pop old crop id=%s on rep update", old_id)

                    # update class1 and class2 representative entries
                    _maybe_update_rep("1", "class1_merged")
                    _maybe_update_rep("2", "class2_merged")
            except Exception:
                logger.exception("FilterManager: error while updating rep-crop mapping in add_event")
            
            logger.debug("FilterManager.add_event: uid=%s has_union_ref=%s", ev.get("uid"), bool(ev.get("_union_crop_ref")))
            ###


    # -----------------------
    # text grouping utilities
    # -----------------------
    def _levenshtein(self, a: str, b: str) -> int:
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        prev = list(range(lb + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i] + [0] * lb
            for j, cb in enumerate(b, start=1):
                add = prev[j] + 1
                delete = cur[j - 1] + 1
                subs = prev[j - 1] + (0 if ca == cb else 1)
                cur[j] = min(add, delete, subs)
            prev = cur
        return prev[lb]

    def _normalize(self, s: str) -> str:
        if s is None:
            return ""
        return self._norm_re.sub("", str(s).upper()).strip()

    def _norm_dist(self, a: str, b: str) -> float:
        if not a and not b:
            return 0.0
        if not a or not b:
            return 1.0
        return self._levenshtein(a, b) / max(1, max(len(a), len(b)))

    def _group_events(self, events: List[dict], key_name: str):
        groups = []
        for ev in events:
            raw = ev.get(key_name) or (ev.get("merged") or {}).get(key_name.replace("_merged", "")) or ""
            norm = self._normalize(raw)
            if not norm:
                continue
            placed = False
            for g in groups:
                if self._norm_dist(g["rep"], norm) <= self.sim_threshold:
                    g["items"].append(ev)
                    g["count"] += 1
                    # update ok_count / best_conf / last_ts
                    if key_name == "class1_merged":
                        ck = ev.get("class1_check") or (ev.get("checks") or {}).get("class1")
                    else:
                        ck = ev.get("class2_check") or (ev.get("checks") or {}).get("class2")
                    if ck and ck.get("status") == "OK":
                        g["ok_count"] += 1
                    try:
                        conf_candidates = []
                        if isinstance(ck, dict):
                            conf_candidates.append(ck.get("confidence") or ck.get("conf"))
                        conf_candidates.append(ev.get("ocr_conf"))
                        for v in conf_candidates:
                            try:
                                if v is not None:
                                    g["best_conf"] = max(g.get("best_conf", 0.0), float(v))
                            except Exception:
                                pass
                    except Exception:
                        pass
                    g["last_ts"] = max(g["last_ts"], ev.get("ts", 0.0))
                    placed = True
                    break
            if not placed:
                # create new group
                if key_name == "class1_merged":
                    ck = ev.get("class1_check") or (ev.get("checks") or {}).get("class1")
                else:
                    ck = ev.get("class2_check") or (ev.get("checks") or {}).get("class2")
                best_conf = 0.0
                try:
                    if isinstance(ck, dict):
                        v0 = ck.get("confidence") or ck.get("conf")
                        if v0 is not None:
                            best_conf = max(best_conf, float(v0))
                except Exception:
                    pass
                try:
                    v1 = ev.get("ocr_conf")
                    if v1 is not None:
                        best_conf = max(best_conf, float(v1))
                except Exception:
                    pass
                groups.append({
                    "rep": norm,
                    "items": [ev],
                    "count": 1,
                    "ok_count": 1 if (ck and ck.get("status") == "OK") else 0,
                    "last_ts": ev.get("ts", 0.0),
                    "best_conf": best_conf,
                })
        return groups

    def _decide_for_key(self, window_events: List[dict], key_name: str, check_key: str) -> Tuple[str, dict, Optional[dict]]:
        """
        Decide representative text for key_name over window_events.
        Returns (raw_text, check_dict, latest_ev) where latest_ev is the event chosen as representative.
        """
        relevant = []
        for r in window_events:
            if r.get(key_name):
                relevant.append(r)
            else:
                merged = r.get("merged") or {}
                k = key_name.replace("_merged", "")
                if merged.get(k):
                    copy = dict(r)
                    copy[key_name] = merged.get(k)
                    relevant.append(copy)
        if not relevant:
            return "", {}, None
        groups = self._group_events(relevant, key_name)
        if not groups:
            return "", {}, None
        total_ok = sum(g["ok_count"] for g in groups)

        if total_ok > 0:
            groups.sort(key=lambda g: (g["ok_count"], g["count"], g["last_ts"]), reverse=True)
            chosen = groups[0]
        else:
            groups.sort(key=lambda g: (g.get("best_conf", 0.0), g["count"], g["last_ts"]), reverse=True)
            chosen = groups[0]

        latest_ev = max(chosen["items"], key=lambda x: x.get("ts", 0.0))
        raw_text = latest_ev.get(key_name) or (latest_ev.get("merged") or {}).get(key_name.replace("_merged", ""), "")
        check = latest_ev.get(check_key, {}) or (latest_ev.get("checks") or {}).get(key_name.replace("_merged", ""), {})
        return raw_text, check, latest_ev

    # -----------------------
    # main loop
    # -----------------------
    def _loop(self):
        while not self._stop.is_set():
            time.sleep(self.interval)
            now = time.time()
            with self._lock:
                window_start = now - self.interval
                window = [e for e in self._events if e.get("ts", 0.0) >= window_start]
                # keep a longer history but prune older events
                self._events = [e for e in self._events if e.get("ts", 0.0) >= now - (self.interval * 10)]
            if not window:
                continue

            # Decide representative texts and obtain events that contributed them
            class1_sel, class1_check, class1_ev = self._decide_for_key(window, "class1_merged", "class1_check")
            class2_sel, class2_check, class2_ev = self._decide_for_key(window, "class2_merged", "class2_check")

            logger.debug(
                "FILTER DEBUG: class1_sel=%r class1_ev_uid=%s class2_sel=%r class2_ev_uid=%s",
                class1_sel, (class1_ev.get("uid") if class1_ev else None),
                class2_sel, (class2_ev.get("uid") if class2_ev else None),
            )

            # helper: extract detections of a class from an event (provenance-tagged)
            def _dets_from_event_for_class(ev: dict, target_class_str: str) -> List[dict]:
                out = []
                if not ev:
                    return out
                dets = ev.get("detections_full") or ev.get("detections") or []
                for d in dets:
                    try:
                        if str(d.get("class")) == str(target_class_str):
                            d2 = dict(d)
                            d2["_event_uid"] = ev.get("uid")
                            d2["_event_ts"] = ev.get("ts", ev.get("timestamp", 0.0))
                            out.append(d2)
                    except Exception:
                        continue
                return out

            # fallback search for most recent event that has detections for a class
            def _find_any_detections_for_class(window_events: List[dict], target_class_str: str) -> List[dict]:
                candidates: List[Tuple[float, dict]] = []
                for ev in window_events:
                    dets = ev.get("detections_full") or ev.get("detections") or []
                    for d in dets:
                        try:
                            if str(d.get("class")) == str(target_class_str):
                                candidates.append((ev.get("ts", 0.0), ev))
                                break
                        except Exception:
                            continue
                if not candidates:
                    return []
                candidates.sort(key=lambda x: x[0])
                chosen_ev = candidates[-1][1]
                return _dets_from_event_for_class(chosen_ev, target_class_str)

            # Collect detections preferring the events that provided the decided texts
            class1_dets = _dets_from_event_for_class(class1_ev, "1") if class1_sel else []
            class2_dets = _dets_from_event_for_class(class2_ev, "2") if class2_sel else []

            # Fallback: if the event that provided the text lacks detections for that class, search window
            if class1_sel and not class1_dets:
                fb = _find_any_detections_for_class(window, "1")
                if fb:
                    class1_dets = fb
                    logger.debug("FilterManager: fallback found %d class1 detections from other event", len(fb))
            if class2_sel and not class2_dets:
                fb = _find_any_detections_for_class(window, "2")
                if fb:
                    class2_dets = fb
                    logger.debug("FilterManager: fallback found %d class2 detections from other event", len(fb))

            # Prefer a single recent event that contains BOTH classes (best case: same-frame union)
            # --- prefer a single event that contains BOTH class1 and class2 detections ---
            primary_candidate_uid = None
            frame_override_event: Optional[dict] = None
            if class1_sel and class2_sel:
                both_event_found = None
                for ev in reversed(window):  # recent first
                    ev_dets = ev.get("detections_full") or ev.get("detections") or []
                    if not ev_dets:
                        continue
                    classes_in_ev = set()
                    for d in ev_dets:
                        try:
                            classes_in_ev.add(str(d.get("class")))
                        except Exception:
                            pass
                    if "1" in classes_in_ev and "2" in classes_in_ev:
                        both_event_found = ev
                        break
                if both_event_found:
                    # Use this event's detections for both classes (provenance-tag them)
                    class1_dets = _dets_from_event_for_class(both_event_found, "1")
                    class2_dets = _dets_from_event_for_class(both_event_found, "2")

                    # IMPORTANT: Do NOT overwrite the previously-decided merged texts or checks.
                    # We keep class1_sel/class2_sel and their checks as selected by grouping logic (respect pipeline/utility).
                    # Only force using this event's detections and frame for the union.
                    frame_override_event = both_event_found
                    primary_candidate_uid = both_event_found.get("uid")

                    logger.debug(
                        "FilterManager: found single-event with both classes uid=%s; using its detections+frame for union (kept merged from grouping: class1=%r class2=%r)",
                        primary_candidate_uid, class1_sel, class2_sel,
                    )

                    # Update per-class detections source provenance so downstream knows dets came from this uid
                    # (detections_source_uid_by_class will be recomputed below from class1_dets/class2_dets)

            # Build chosen_detections union (provenance-tagged)
            chosen_detections: List[dict] = []
            if class1_dets:
                chosen_detections.extend(class1_dets)
            if class2_dets:
                chosen_detections.extend(class2_dets)

            # Recompute uid counts from the chosen_detections
            det_uids_counts: Dict[Any, int] = {}
            for d in chosen_detections:
                uid = d.get("_event_uid")
                if uid:
                    det_uids_counts[uid] = det_uids_counts.get(uid, 0) + 1

            # Determine primary_dets_uid (prefer primary_candidate from both-event)
            primary_dets_uid = None
            if primary_candidate_uid:
                primary_dets_uid = primary_candidate_uid
            elif det_uids_counts:
                try:
                    primary_dets_uid = max(
                        det_uids_counts.keys(),
                        key=lambda uid: (
                            det_uids_counts.get(uid, 0),
                            max([d.get("_event_ts", 0.0) for d in chosen_detections if d.get("_event_uid") == uid]),
                        ),
                    )
                except Exception:
                    primary_dets_uid = next(iter(det_uids_counts.keys()))

            # If primary uid covers the majority, restrict union to that uid to avoid mixing frames
            if det_uids_counts and primary_dets_uid:
                total_dets = sum(det_uids_counts.values())
                primary_count = det_uids_counts.get(primary_dets_uid, 0)
                if primary_count >= max(1, int(0.6 * total_dets)):  # threshold tunable
                    chosen_detections = [d for d in chosen_detections if d.get("_event_uid") == primary_dets_uid]
                    logger.debug(
                        "FilterManager: using primary_dets_uid=%s for union (count %d of %d)",
                        primary_dets_uid, primary_count, total_dets,
                    )
                else:
                    logger.debug(
                        "FilterManager: mixed det uids %s (primary %s count=%d)",
                        det_uids_counts, primary_dets_uid, primary_count,
                    )

            # Frame selection: prefer explicit frame_override_event, then primary_dets_uid event, then other fallbacks
            chosen_frame_event: Optional[dict] = None
            if frame_override_event:
                chosen_frame_event = frame_override_event
            else:
                if primary_dets_uid:
                    ev_match = next((r for r in reversed(window) if r.get("uid") == primary_dets_uid), None)
                    if ev_match and (ev_match.get("frame_b64") or ev_match.get("image_b64")):
                        chosen_frame_event = ev_match

            if not chosen_frame_event:
                both_candidates = [r for r in window if (r.get("frame_b64") or r.get("image_b64")) and (r.get("detections_full") or r.get("detections"))]
                if both_candidates:
                    match = None
                    for r in reversed(both_candidates):
                        if any(d.get("_event_uid") == r.get("uid") for d in chosen_detections if d.get("_event_uid")):
                            match = r
                            break
                    chosen_frame_event = match or max(both_candidates, key=lambda r: r.get("ts", 0.0))

            if not chosen_frame_event:
                frame_candidates = [r for r in window if r.get("frame_b64") or r.get("image_b64")]
                if frame_candidates:
                    chosen_frame_event = max(frame_candidates, key=lambda r: r.get("ts", 0.0))

            # We no longer propagate full-frame base64 (frame_b64). Keep image_b64 (annotated) if present.
            last_img_b64 = chosen_frame_event.get("image_b64") if chosen_frame_event else ""
            frame_src_uid = chosen_frame_event.get("uid") if chosen_frame_event else None
            # --- ALIGN merged/checks TO CHOSEN FRAME EVENT (to guarantee text/image/check provenance match) ---
            # Assumes chosen_frame_event and frame_src_uid have been determined already.

            # Preserve original selections in merged_values_by_uid for audit/tracing
            merged_values_by_uid = {}
            try:
                # record merged from events we used to select text
                for ev in (class1_ev, class2_ev, frame_override_event):
                    if ev and isinstance(ev, dict):
                        if ev.get("uid") and ev.get("merged"):
                            merged_values_by_uid[ev.get("uid")] = ev.get("merged")
            except Exception:
                merged_values_by_uid = {}

            # If we have a chosen frame event, prefer its merged/checks for final payload,
            # so that logged merged, checks and saved image all originate from the same event.
            if chosen_frame_event and isinstance(chosen_frame_event, dict):
                try:
                    ev_frame = chosen_frame_event
                    ev_uid = ev_frame.get("uid")
                    ev_merged = ev_frame.get("merged") or {}
                    ev_checks = ev_frame.get("checks") or {}

                    # Only overwrite the selected merged/checks for classes present in the frame event.
                    # If the frame event lacks merged for a class, keep prior selection.
                    overwritten = False
                    if ev_merged:
                        # class1
                        c1_val = (ev_merged.get("class1") or "").strip()
                        if c1_val:
                            class1_sel = c1_val
                            # prefer the event's checks if present
                            if isinstance(ev_checks, dict) and ev_checks.get("class1"):
                                class1_check = ev_checks.get("class1")
                            # make provenance point to this event
                            class1_ev = ev_frame
                            overwritten = True

                        # class2
                        c2_val = (ev_merged.get("class2") or "").strip()
                        if c2_val:
                            class2_sel = c2_val
                            if isinstance(ev_checks, dict) and ev_checks.get("class2"):
                                class2_check = ev_checks.get("class2")
                            class2_ev = ev_frame
                            overwritten = True

                    # Record the frame's merged for tracing as well
                    if ev_uid and ev_merged:
                        merged_values_by_uid.setdefault(ev_uid, ev_merged)
                    if overwritten:
                        logger.debug(
                            "FilterManager: aligned merged/checks to chosen frame event uid=%s (class1=%r class2=%r)",
                            ev_uid, class1_sel, class2_sel
                        )
                except Exception:
                    logger.exception("FilterManager: error while aligning merged/checks to chosen frame event")

            # Build merged_source_uid_by_class from class*_ev (after possible alignment)
            merged_source_uid_by_class = {
                "1": (class1_ev.get("uid") if class1_ev and isinstance(class1_ev, dict) else None),
                "2": (class2_ev.get("uid") if class2_ev and isinstance(class2_ev, dict) else None),
            }

            # recompute detections provenance summary (ensure variables exist)
            detections_by_class = {"1": len(class1_dets or []), "2": len(class2_dets or [])}
            detections_source_uid_by_class = {
                "1": next((d.get("_event_uid") for d in (class1_dets or []) if d.get("_event_uid")), None),
                "2": next((d.get("_event_uid") for d in (class2_dets or []) if d.get("_event_uid")), None),
            }
            detections_source_uid = primary_dets_uid or detections_source_uid_by_class.get("1") or detections_source_uid_by_class.get("2")

            # mixed_sources: whether original merged sources differ from frame source (kept for audit)
            mixed_sources = False
            try:
                if frame_src_uid:
                    if merged_source_uid_by_class.get("1") and merged_source_uid_by_class.get("1") != frame_src_uid:
                        mixed_sources = True
                    if merged_source_uid_by_class.get("2") and merged_source_uid_by_class.get("2") != frame_src_uid:
                        mixed_sources = True
            except Exception:
                mixed_sources = False

            # If frame and detections come from different events, log a warning (audit)
            if frame_src_uid and detections_source_uid and frame_src_uid != detections_source_uid:
                logger.warning(
                    "FilterManager: frame and detections come from different events (frame_uid=%s dets_uid=%s) window_ts=%s",
                    frame_src_uid, detections_source_uid, now,
                )

            # Build final filtered_payload: merged/checks now aligned to chosen_frame_event when available,
            # and we still include merged_values_by_uid & provenance for later audit.
            filtered_payload = {
                "timestamp": now,
                "ts": now,
                "uid": "filtered",
                "detections": chosen_detections,
                "merged": {"class1": class1_sel, "class2": class2_sel},
                "checks": {"class1": class1_check or {}, "class2": class2_check or {}},
                "image_b64": last_img_b64 or "",
                "filtered": True,
                "filter_interval": float(self.interval),
                # provenance
                "frame_source_uid": frame_src_uid,
                "detections_source_uid": detections_source_uid,
                "detections_by_class": detections_by_class,
                "detections_source_uid_by_class": detections_source_uid_by_class,
                "merged_source_uid_by_class": merged_source_uid_by_class,
                "mixed_sources": mixed_sources,
                "merged_values_by_uid": merged_values_by_uid,
            }
            ###
            # Decide which cached union-crop id to attach (if any).
            try:
                chosen_crop_id = None
                # prefer chosen_frame_event's crop (this ensures frame provenance is preserved)
                if chosen_frame_event and isinstance(chosen_frame_event, dict):
                    cref = chosen_frame_event.get("_union_crop_ref") or chosen_frame_event.get("_chosen_union_crop_ref")
                    if isinstance(cref, dict) and cref.get("id"):
                        chosen_crop_id = cref.get("id")
                # else prefer representative crop by signature (class1 then class2)
                if not chosen_crop_id:
                    # normalized signatures
                    sig1_norm = self._normalize(class1_sel) if class1_sel else ""
                    sig2_norm = self._normalize(class2_sel) if class2_sel else ""
                    # try class1 rep
                    if sig1_norm:
                        ent = self._rep_crop_by_sig.get(("1", sig1_norm))
                        if ent and ent.get("id"):
                            chosen_crop_id = ent.get("id")
                    # try class2 if still none
                    if not chosen_crop_id and sig2_norm:
                        ent2 = self._rep_crop_by_sig.get(("2", sig2_norm))
                        if ent2 and ent2.get("id"):
                            chosen_crop_id = ent2.get("id")

                # verify presence in cache (best-effort); if present, attach chosen ref to payload
                if chosen_crop_id:
                    try:
                        from app.utils import crop_cache
                        if crop_cache.contains(chosen_crop_id):
                            filtered_payload["_chosen_union_crop_ref"] = {"type": "mem", "id": chosen_crop_id}
                        else:
                            # cache miss: remove any stale mapping entries that point to this id
                            # cleanup rep map entries referring to missing id
                            for k, v in list(self._rep_crop_by_sig.items()):
                                if v and v.get("id") == chosen_crop_id and not crop_cache.contains(chosen_crop_id):
                                    try:
                                        del self._rep_crop_by_sig[k]
                                    except Exception:
                                        pass
                    except Exception:
                        logger.exception("FilterManager: error while validating/attaching chosen crop id=%s", chosen_crop_id)
            except Exception:
                logger.exception("FilterManager: error while selecting chosen crop to attach")
            ###


            # 예: chosen_frame_event는 FilterManager가 union에 선택한 프레임 이벤트 객체입니다.
            # filtered_payload 생성 후 바로:
            try:
                # ensure we attach the original full-frame base64 for downstream consumers (ResultManager)
                if chosen_frame_event and chosen_frame_event.get("frame_b64"):
                    filtered_payload["frame_b64"] = chosen_frame_event.get("frame_b64")
                else:
                    # optional: if chosen_frame_event isn't available but frame_src (uid) is known,
                    # try to find the corresponding event in the current window/list and attach its frame_b64.
                    # (only if you keep an accessible list of recent events)
                    pass
            
            except Exception:
                logger.exception("Failed to attach frame_b64 to filtered_payload for uid=%s", filtered_payload.get("uid"))
                        # debug emit payload summary for traceability
            try:
                logger.debug(
                    "EMIT filtered_payload uid=%s frame_src=%s dets_src=%s dets_by_class=%s sample_dets=%s",
                    filtered_payload.get("uid"),
                    filtered_payload.get("frame_source_uid"),
                    filtered_payload.get("detections_source_uid"),
                    filtered_payload.get("detections_by_class"),
                    [(d.get("_event_uid"), d.get("class"), d.get("bbox")) for d in filtered_payload.get("detections", [])[:6]],
                )
            except Exception:
                pass
            try:
                logger.debug("DEBUG FilterManager emitting payload keys uid=%s keys=%s frame_b64_len=%d",
                            filtered_payload.get("uid"),
                            list(filtered_payload.keys()),
                            len(filtered_payload.get("frame_b64") or ""))
            except Exception:
                pass


            # --- EMIT decision: use class1_text + class1_check.status as signature (no frame uid) ---
            try:
                sig = (
                    (class1_sel or ""),
                    (class1_check or {}).get("status", "") if isinstance(class1_check, dict) else "",
                )
            except Exception:
                sig = ((class1_sel or ""),)

            should_emit = True
            try:
                if self._last_emitted_sig is not None and sig == self._last_emitted_sig:
                    elapsed = now - float(self._last_emit_ts or 0.0)
                    if elapsed < float(self.suppress_repeat_seconds):
                        should_emit = False
            except Exception:
                should_emit = True

            logger.debug(
                "EMIT_DECISION: sig=%s last_sig=%s elapsed=%.2f should_emit=%s merged=(%s,%s) frame_src=%s dets_src=%s",
                sig,
                self._last_emitted_sig,
                now - float(self._last_emit_ts or 0.0),
                should_emit,
                class1_sel, class2_sel,
                frame_src_uid,
                detections_source_uid,
            )

            if should_emit:
                try:
                    self.emit_fn(filtered_payload)
                except Exception:
                    logger.exception("FilterManager.emit_fn raised an exception")
                else:
                    self._last_emitted_sig = sig
                    self._last_emit_ts = now


    def run(self):
        self._loop()

    def start_and_run(self):
        self.start()