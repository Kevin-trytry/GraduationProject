# -*- coding: utf-8 -*-
"""
elevator_input.py — 手勢 → 兩位數樓層 的輸入狀態機

設計原則
────────
1. **純邏輯，不做任何 I/O。** `update()` 吃「現在時間 + 這一幀的偵測框」，
   吐出事件 list。這樣可以用既有的 detections_log.csv 回放測試，
   不需要板子、不需要相機。

2. **時間驅動，不是幀數驅動。** 實測幀率在 3.5~4.6 fps 之間浮動，
   而且出現過 4 秒的空檔。所有視窗一律用秒數算。

3. **拒絕要看得見。** 停留時間到了但品質不夠時，會發出 digit_rejected
   事件並附上統計數字。不然使用者只會看到「比半天沒反應」而不知道為什麼。

輸入流程
────────
    比出數字，維持 3 秒      → 鎖定第 1 位
    手放下 0.8 秒            → 分隔
    比出數字，維持 3 秒      → 鎖定第 2 位
    手放下 0.8 秒            → 進入確認
    3 秒內不要舉手           → 送出樓層
    （確認期間舉手 = 取消）

為什麼要「掐頭去尾」
──────────────────
手勢剛成形和要放下的那零點幾秒，手指還在移動，是誤判最多的時段。
3 秒的停留裡只取中間 2 秒（約 8~10 幀）做多數決，把那些過渡幀丟掉。

樓層編碼
────────
    第 1 位 ∈ {0~9, B}，第 2 位 ∈ {0~9}
    B + n  →  地下 n 樓（B1~B9）
    0 + n  →  n 樓（1F~9F）
    m + n  →  mn 樓（10F~99F）
    不合法：00、B0、第 2 位比 B
"""

from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import List, Optional

B_CLASS = 10          # 韌體的類別索引：0~9 是數字，10 是 B
DIGIT_CLASSES = set(range(10))


# ─── 參數 ────────────────────────────────────────────────────────────────────
@dataclass
class ElevatorConfig:
    # 停留鎖定
    dwell: float = 3.0            # 一個數字要維持幾秒
    trim_head: float = 0.5        # 丟掉開頭幾秒（手勢成形中）
    trim_tail: float = 0.5        # 丟掉結尾幾秒（準備放下）
    min_frames: int = 5           # 中間視窗至少要有幾幀才算數
    agree_ratio: float = 0.70     # 視窗內多數決的最低一致比例
    conf_min: float = 30.0        # 勝出類別的信心度中位數下限（%）

    # 流程
    gap: float = 0.8              # 兩位數之間，手要離開多久才算分隔
    confirm: float = 3.0          # 兩位齊了之後的確認等待時間
    idle_timeout: float = 15.0    # 閒置多久自動重置

    # 偵測門檻
    conf_floor: float = 20.0      # 低於此信心度的框直接當雜訊丟掉（與韌體一致）

    # 樓層範圍（None = 不限）
    min_floor: Optional[int] = 1
    max_floor: Optional[int] = 99
    max_basement: Optional[int] = 9

    # 事件節流
    progress_interval: float = 0.25
    reject_cooldown: float = 1.0


@dataclass
class _Sample:
    t: float
    cls: int
    conf: float


# ─── 樓層編碼 ────────────────────────────────────────────────────────────────
def format_floor(d1: int, d2: int, cfg: ElevatorConfig):
    """回傳 (顯示字串, 結構化值) 或 (None, 失敗原因)。"""
    if d2 == B_CLASS:
        return None, 'B 只能放在第一位（B1 = 地下一樓）'
    if d1 == B_CLASS:
        if d2 == 0:
            return None, '沒有 B0 這個樓層'
        if cfg.max_basement is not None and d2 > cfg.max_basement:
            return None, f'地下室只到 B{cfg.max_basement}'
        return f'B{d2}', {'kind': 'basement', 'level': d2, 'number': -d2}
    n = d1 * 10 + d2
    if n == 0:
        return None, '沒有 0 樓'
    if cfg.min_floor is not None and n < cfg.min_floor:
        return None, f'最低樓層是 {cfg.min_floor}F'
    if cfg.max_floor is not None and n > cfg.max_floor:
        return None, f'最高樓層是 {cfg.max_floor}F'
    return f'{n}F', {'kind': 'above', 'level': n, 'number': n}


# ─── 狀態機 ──────────────────────────────────────────────────────────────────
class ElevatorInput:
    """
    狀態：
        idle      等待第一個手勢
        holding   正在累積停留時間
        gapwait   數字已鎖定，等手離開
        confirm   兩位齊了，等確認（此時舉手 = 取消）
    """

    def __init__(self, cfg: Optional[ElevatorConfig] = None):
        self.cfg = cfg or ElevatorConfig()
        self.reset(0.0, emit=False)

    # ── 對外 ────────────────────────────────────────────────────────────────
    def reset(self, now: float, emit: bool = True, reason: str = 'reset'):
        self.state = 'idle'
        self.digits: List[int] = []
        self.samples: List[_Sample] = []
        self.hold_start: Optional[float] = None
        self.last_det_t: Optional[float] = None
        self.last_any_t: float = now
        self.gap_start: Optional[float] = None
        self.confirm_start: Optional[float] = None
        self._last_progress = 0.0
        self._last_reject = -999.0
        self.pending = None
        return [self._ev(now, 'reset', reason=reason)] if emit else []

    def snapshot(self, now: float) -> dict:
        """給 UI / HTTP 用的當前狀態（不改變任何東西）。"""
        s = {
            'state': self.state,
            'digits': [_name(d) for d in self.digits],
            'pending_floor': self.pending[0] if self.pending else None,
            'pos': len(self.digits) + 1,
        }
        if self.state == 'holding' and self.hold_start is not None:
            el = now - self.hold_start
            s['hold_elapsed'] = round(el, 2)
            s['hold_need'] = self.cfg.dwell
            s['hold_ratio'] = round(min(el / self.cfg.dwell, 1.0), 3)
            if self.samples:
                s['holding'] = _name(self.samples[-1].cls)
        if self.state == 'gapwait':
            s['gap_need'] = self.cfg.gap
            s['gap_elapsed'] = (round(now - self.gap_start, 2)
                                if self.gap_start is not None else 0.0)
            s['gap_ratio'] = round(min(s['gap_elapsed'] / self.cfg.gap, 1.0), 3)
            s['hand_present'] = self.gap_start is None
        if self.state == 'confirm' and self.confirm_start is not None:
            s['confirm_remain'] = round(
                max(0.0, self.cfg.confirm - (now - self.confirm_start)), 2)
        return s

    def update(self, now: float, dets) -> List[dict]:
        """
        now  : 單調遞增的秒數（time.monotonic() 或回放時的相對時間）
        dets : 這一幀的偵測框，格式 [x, y, w, h, score(0~100), cls_id]
        """
        events: List[dict] = []
        top = self._pick(dets)

        if top is not None:
            self.samples.append(_Sample(now, top[0], top[1]))
            self.last_det_t = now
            self.last_any_t = now
            # 只留下最近 dwell + 1 秒的樣本，避免無限成長
            cut = now - (self.cfg.dwell + 1.0)
            if self.samples and self.samples[0].t < cut:
                self.samples = [s for s in self.samples if s.t >= cut]

        handler = getattr(self, f'_st_{self.state}')
        events += handler(now, top)

        # 閒置逾時（確認階段不算，它有自己的計時）
        if self.state not in ('idle', 'confirm') and \
                now - self.last_any_t > self.cfg.idle_timeout:
            events += self.reset(now, reason='idle_timeout')

        return events

    # ── 各狀態 ──────────────────────────────────────────────────────────────
    def _st_idle(self, now, top):
        if top is None:
            return []
        self.state = 'holding'
        self.hold_start = now
        self.samples = [_Sample(now, top[0], top[1])]
        return [self._ev(now, 'hold_started', pos=len(self.digits) + 1)]

    def _st_holding(self, now, top):
        ev = []
        # 手離開太久 → 這次停留作廢
        if self.last_det_t is None or now - self.last_det_t > self.cfg.gap:
            if self.digits:
                # 已經鎖過第一位，退回等待分隔（第一位保留）
                self.state = 'gapwait'
                self.gap_start = self.last_det_t or now
                return ev
            return ev + self.reset(now, reason='hand_left_during_hold')

        elapsed = now - self.hold_start
        if elapsed < self.cfg.dwell:
            if now - self._last_progress >= self.cfg.progress_interval:
                self._last_progress = now
                ev.append(self._ev(
                    now, 'progress',
                    pos=len(self.digits) + 1,
                    holding=_name(top[0]) if top else None,
                    elapsed=round(elapsed, 2),
                    need=self.cfg.dwell,
                    ratio=round(min(elapsed / self.cfg.dwell, 1.0), 3),
                ))
            return ev

        # 停留時間到，評估中間視窗
        ok, info = self._evaluate(now)
        if not ok:
            if now - self._last_reject >= self.cfg.reject_cooldown:
                self._last_reject = now
                ev.append(self._ev(now, 'digit_rejected',
                                   pos=len(self.digits) + 1, **info))
            # 重新開始累積，使用者繼續維持就會再試一次
            self.hold_start = now
            self.samples = [s for s in self.samples if s.t >= now - 0.1]
            return ev

        self.digits.append(info['value_cls'])
        ev.append(self._ev(now, 'digit_locked',
                           pos=len(self.digits),
                           value=_name(info['value_cls']),
                           **{k: v for k, v in info.items() if k != 'value_cls'}))
        self.state = 'gapwait'
        self.gap_start = None
        self.samples = []
        return ev

    def _st_gapwait(self, now, top):
        if top is not None:          # 手還在，分隔重新計時
            self.gap_start = None
            if now - self._last_progress >= self.cfg.progress_interval:
                self._last_progress = now
                return [self._ev(now, 'gap_progress', hand_present=True,
                                 elapsed=0.0, need=self.cfg.gap,
                                 hint='把手移開才能輸入下一位')]
            return []
        if self.gap_start is None:
            self.gap_start = now
            return []
        if now - self.gap_start < self.cfg.gap:
            if now - self._last_progress >= self.cfg.progress_interval:
                self._last_progress = now
                return [self._ev(now, 'gap_progress', hand_present=False,
                                 elapsed=round(now - self.gap_start, 2),
                                 need=self.cfg.gap)]
            return []

        # 分隔完成
        if len(self.digits) >= 2:
            self.state = 'confirm'
            self.confirm_start = now
            name, val = format_floor(self.digits[0], self.digits[1], self.cfg)
            if name is None:
                ev = [self._ev(now, 'floor_invalid',
                               digits=[_name(d) for d in self.digits],
                               reason=val)]
                return ev + self.reset(now, reason='invalid_floor')
            self.pending = (name, val)
            return [self._ev(now, 'floor_pending', floor=name, value=val,
                             digits=[_name(d) for d in self.digits],
                             confirm_sec=self.cfg.confirm)]

        self.state = 'idle'
        self.samples = []
        return [self._ev(now, 'ready_for_next', pos=len(self.digits) + 1)]

    def _st_confirm(self, now, top):
        if top is not None:
            ev = [self._ev(now, 'cancelled', reason='hand_raised_during_confirm',
                           floor=self.pending[0] if self.pending else None)]
            return ev + self.reset(now, reason='cancelled')
        remain = self.cfg.confirm - (now - self.confirm_start)
        if remain > 0:
            if now - self._last_progress >= self.cfg.progress_interval:
                self._last_progress = now
                return [self._ev(now, 'confirm_progress',
                                 floor=self.pending[0],
                                 remain=round(remain, 2))]
            return []
        name, val = self.pending
        ev = [self._ev(now, 'floor_dispatched', floor=name, value=val,
                       digits=[_name(d) for d in self.digits])]
        return ev + self.reset(now, emit=False)

    # ── 內部 ────────────────────────────────────────────────────────────────
    def _pick(self, dets):
        """取這一幀信心度最高、且過得了門檻的框 → (cls, conf)。"""
        if not dets:
            return None
        best = None
        for d in dets:
            if len(d) < 6:
                continue
            conf = float(d[4])
            cls = int(d[5])
            if conf < self.cfg.conf_floor:
                continue
            if cls not in DIGIT_CLASSES and cls != B_CLASS:
                continue
            if best is None or conf > best[1]:
                best = (cls, conf)
        return best

    def _evaluate(self, now):
        """對 [hold_start+trim_head, hold_start+dwell-trim_tail] 做多數決。"""
        c = self.cfg
        lo = self.hold_start + c.trim_head
        hi = self.hold_start + c.dwell - c.trim_tail
        win = [s for s in self.samples if lo <= s.t <= hi]

        if len(win) < c.min_frames:
            return False, {'reason': 'frames_too_few',
                           'frames': len(win), 'need_frames': c.min_frames,
                           'detail': f'中間 {hi - lo:.1f} 秒只收到 {len(win)} 幀，'
                                     f'至少要 {c.min_frames} 幀'}

        counts = Counter(s.cls for s in win)
        top_cls, n = counts.most_common(1)[0]
        agree = n / len(win)
        if agree < c.agree_ratio:
            return False, {'reason': 'agreement_low',
                           'frames': len(win), 'agree': round(agree, 3),
                           'need_agree': c.agree_ratio,
                           'top': _name(top_cls),
                           'spread': {_name(k): v for k, v in counts.items()},
                           'detail': f'{len(win)} 幀裡只有 {n} 幀是 {_name(top_cls)}'
                                     f'（{agree:.0%}），手勢不夠穩定'}

        confs = [s.conf for s in win if s.cls == top_cls]
        cmed = median(confs)
        if cmed < c.conf_min:
            return False, {'reason': 'confidence_low',
                           'frames': len(win), 'agree': round(agree, 3),
                           'top': _name(top_cls),
                           'conf_median': round(cmed, 1),
                           'need_conf': c.conf_min,
                           'detail': f'{_name(top_cls)} 的信心度中位數只有 '
                                     f'{cmed:.0f}%，低於 {c.conf_min:.0f}%'}

        return True, {'value_cls': top_cls, 'frames': len(win),
                      'agree': round(agree, 3),
                      'conf_median': round(cmed, 1),
                      'spread': {_name(k): v for k, v in counts.items()}}

    def _ev(self, now, event, **kw):
        d = {'t': round(now, 3), 'event': event, 'state': self.state}
        if self.digits:
            d['digits'] = [_name(x) for x in self.digits]
        d.update(kw)
        return d


def _name(cls: Optional[int]) -> Optional[str]:
    if cls is None:
        return None
    return 'B' if cls == B_CLASS else str(cls)
