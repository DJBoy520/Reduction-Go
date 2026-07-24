"""
ML-DSA 格攻击核心引擎。

从 api.py 提取的攻击编排逻辑，实现:
  - 选择攻击策略（签名 / 证书）
  - 从密钥/证书提取攻击材料
  - 调用格约减并组装结果
  - 保留 _LatticeAttackResult 到 dict 的序列化路径

设计原则:
  - 不做日志全局配置
  - 不做 sys.exit()
  - 返回结构化 AttackResult (dict)
  - 支持可选进度回调
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .materials import AttackMaterial, CertMaterial, SyntheticMaterial
from . import strategy
from ..progress import (
    AttackStage,
    ProgressEvent,
    _LatticeAttackResult,
    _print_stage_start,
    _print_stage_done,
    _print_progress,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结果组装
# ---------------------------------------------------------------------------

def _serialize_result(res: _LatticeAttackResult) -> dict:
    """将 _LatticeAttackResult 转为可序列化的 dict。"""
    # 兼容旧版 _LatticeAttackResult 可能有的字段
    def _get(name, default=None):
        return getattr(res, name, default)

    basis = _get("basis_reduced")
    basis_raw = _get("basis_raw")
    privkey_bytes = _get("privkey_bytes")

    out: Dict[str, Any] = {
        "success": bool(_get("found")),
        "timed_out": bool(_get("timed_out")),
        "attack_method": _get("attack_method"),
        "elapsed_s": _get("elapsed_s"),
        "attack_material": _get("attack_material"),
        "e_total": _get("e_total"),
        "e_norm": _get("e_norm"),
        "recovered_secret_norm": _get("recovered_secret_norm"),
        "recovery_method": _get("recovery_method"),
        "recovery_confidence": _get("recovery_confidence"),
        "sk_hex": _get("sk_hex"),
    }

    # 可选字段
    if _get("lattice_dim"):
        out["lattice_dim"] = _get("lattice_dim")
    if _get("reduction_time_ms"):
        out["reduction_time_ms"] = _get("reduction_time_ms")
    if _get("basis_path"):
        out["basis_path"] = _get("basis_path")

    if basis is not None:
        out["basis_reduced"] = _np_to_list(basis)
    if basis_raw is not None:
        out["basis_raw"] = _np_to_list(basis_raw)
    if privkey_bytes is not None:
        out["sk_b64"] = base64.b64encode(bytes(privkey_bytes)).decode("ascii")
        out["sk_hex"] = _get("sk_hex") or bytes(privkey_bytes).hex()

    return out


def _np_to_list(obj):
    """递归把 numpy array 转成 list，方便 JSON 序列化。"""
    try:
        return obj.tolist()
    except AttributeError:
        pass
    if isinstance(obj, (list, tuple)):
        return [_np_to_list(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# 攻击引擎
# ---------------------------------------------------------------------------

def run_attack(
    config: dict,
    progress: Optional[Callable[[ProgressEvent], None]] = None,
) -> dict:
    """
    运行 ML-DSA 格攻击。

    参数:
        config: AttackConfig dict（由 AttackConfig 转换而来）
        progress: 可选进度回调

    返回:
        AttackResult dict
    """
    try:
        start = time.time()

        # --- 阶段 1: 策略选择 ---
        _print_stage_start(AttackStage.PREPARE, progress, logger)
        strategy_name = strategy.choose_attack_strategy(config)
        config["strategy"] = strategy_name
        _print_stage_done(AttackStage.PREPARE, progress, logger)

        # --- 阶段 2: 攻击材料提取 ---
        _print_stage_start(AttackStage.EXTRACT, progress, logger)
        mat = _extract_material(config, progress)
        _print_stage_done(AttackStage.EXTRACT, progress, logger)

        # --- 阶段 3: 格约减 ---
        _print_stage_start(AttackStage.REDUCE, progress, logger)
        res = _reduce_lattice(config, mat, strategy_name, progress)
        _print_stage_done(AttackStage.REDUCE, progress, logger)

        # --- 阶段 4: 组装结果 ---
        _print_stage_start(AttackStage.POSTPROCESS, progress, logger)
        res = _postprocess(config, mat, res, start)
        _print_stage_done(AttackStage.POSTPROCESS, progress, logger)

        return _serialize_result(res)

    except Exception as e:
        logger.exception("Attack failed")
        return {
            "success": False,
            "error": str(e),
            "timed_out": False,
            "elapsed_s": 0,
            "attack_method": "none",
        }


# ---------------------------------------------------------------------------
# 内部编排
# ---------------------------------------------------------------------------

def _extract_material(
    config: dict,
    progress: Optional[Callable[[ProgressEvent], None]],
) -> AttackMaterial:
    """从密钥或证书提取攻击材料。"""
    if config.get("cert_path"):
        cert_path = config["cert_path"]
        with open(cert_path, "rb") as fh:
            pem_data = fh.read()
        mat = CertMaterial.from_pem(pem_data, logger=logger)
    else:
        mat = SyntheticMaterial.from_config(config, logger=logger)

    _print_progress(
        AttackStage.EXTRACT,
        progress,
        logger,
        f"e_total={mat.e_total}, e_norm={mat.e_norm:.4f}",
    )
    return mat


def _reduce_lattice(
    config: dict,
    mat: AttackMaterial,
    strategy_name: str,
    progress: Optional[Callable[[ProgressEvent], None]],
) -> _LatticeAttackResult:
    """调用格约减引擎。"""
    # 延迟导入，避免循环
    from ..lattice_attack import lattice_attack

    if isinstance(mat, CertMaterial):
        return lattice_attack(
            mat.tbs_der,
            logger=logger,
            progress=progress,
            strategy=strategy_name,
            **{k: v for k, v in config.items() if k not in ("cert_path", "strategy")},
        )
    else:
        # SyntheticMaterial
        from ..crypto.sign import sign_with_seed
        _, sig, _ = sign_with_seed(
            config["sk_seed"],
            config["msg"],
            config.get("d", ""),
            config.get("randomizer", ""),
        )
        return lattice_attack(
            config["msg"],
            sig,
            logger=logger,
            progress=progress,
            strategy=strategy_name,
            **{k: v for k, v in config.items() if k not in ("sk_seed", "msg", "strategy")},
        )


def _postprocess(
    config: dict,
    mat: AttackMaterial,
    res: _LatticeAttackResult,
    start: float,
) -> _LatticeAttackResult:
    """后处理：保存基、组装最终结果。"""
    res.elapsed_s = time.time() - start

    # 保存基矩阵
    basis_path = config.get("save_basis")
    if basis_path and res.basis_reduced is not None:
        import numpy as np
        np.savez(basis_path, basis=res.basis_reduced)
        res.basis_path = os.path.abspath(basis_path)
        _print_progress(
            AttackStage.POSTPROCESS,
            None,
            logger,
            f"saved basis -> {basis_path}",
        )

    return res
