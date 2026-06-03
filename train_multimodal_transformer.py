
from __future__ import annotations

import argparse
import os
import sys
from typing import NoReturn

_here = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_here, ".."))
_IMPL_MODULE = "train_multimodal_transformer_impl"
_IMPL_PATH = os.path.join(_here, f"{_IMPL_MODULE}.py")

_DEFAULT_EEG_NPZ = os.path.join(_REPO_ROOT, "multimodal_overlap", "eeg_numpy_efficientnet_feats.npz")
_DEFAULT_BERT_NPZ = os.path.join(_REPO_ROOT, "multimodal_overlap", "save_audio_ASR", "bert_feats_keyed.npz")
_DEFAULT_SPEECH_NPZ = os.path.join(_REPO_ROOT, "Data_preprocessed_EEGSpeech", "fused_five_modalities.npz")

_PREVIEW_NOTICE = """
[预览版] 当前仓库未包含完整训练实现。

论文发表前暂不公开：
  - 跨被试 ID 共享空间投影与匈牙利最优配对
  - StepBertSpeechFusion + FusionSeqTransformer 时序融合与 hybrid readout
  - 完整训练循环（Focal / EMA / 温度校准等默认配置）

本地运行：在同目录放置 train_multimodal_transformer_impl.py 后执行
  python train_multimodal_transformer.py [与原先相同的参数...]

论文接收后将发布完整代码与复现说明。
""".strip()


def _load_full_implementation():
    if not os.path.isfile(_IMPL_PATH):
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location(_IMPL_MODULE, _IMPL_PATH)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _preview_exit(message: str, code: int = 2) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _build_preview_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="三模态 npz 被试级融合训练（预览入口；完整实现见 impl 或论文代码发布）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_PREVIEW_NOTICE,
    )
    p.add_argument("--eeg-npz", type=str, default=_DEFAULT_EEG_NPZ)
    p.add_argument("--bert-npz", type=str, default=_DEFAULT_BERT_NPZ)
    p.add_argument("--speech-npz", type=str, default=_DEFAULT_SPEECH_NPZ)
    p.add_argument("--no-cross-id-pairs", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join(_REPO_ROOT, "checkpoints", "multimodal_transformer_subjectsplit_best.pth"),
    )
    p.add_argument("--show-withheld", action="store_true", help="列出暂未公开的组件说明后退出。")
    return p


def _run_preview_mode(argv: list[str]) -> None:
    parser = _build_preview_parser()
    if not argv or "-h" in argv or "--help" in argv:
        parser.print_help()
        print()
        print(_PREVIEW_NOTICE)
        if os.path.isfile(_IMPL_PATH):
            print()
            print(f"检测到本地完整实现: {_IMPL_PATH}")
            print("训练请直接带参数运行本脚本（将自动委托 impl），或: python train_multimodal_transformer_impl.py")
        return

    args = parser.parse_args(argv)
    if args.show_withheld:
        print(_PREVIEW_NOTICE)
        print()
        print("数据默认路径（与完整版一致）：")
        print(f"  EEG:    {args.eeg_npz}")
        print(f"  BERT:   {args.bert_npz}")
        print(f"  Speech: {args.speech_npz}")
        return

    _preview_exit(
        _PREVIEW_NOTICE + "\n\n未找到 " + _IMPL_PATH + "。\n请将完整实现放在该路径后再训练。"
    )


def main() -> None:
    argv = sys.argv[1:]

    # 不加载 impl：仅展示说明 / 帮助
    if "--show-withheld" in argv:
        _run_preview_mode(argv)
        return
    if (not argv or "-h" in argv or "--help" in argv) and not os.path.isfile(_IMPL_PATH):
        _run_preview_mode(argv)
        return

    if os.path.isfile(_IMPL_PATH):
        impl = _load_full_implementation()
        if impl is not None and hasattr(impl, "main"):
            impl.main()
            return

    _run_preview_mode(argv)


if __name__ == "__main__":
    main()
