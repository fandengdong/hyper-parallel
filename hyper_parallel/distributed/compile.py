# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Compile Transformer decoder layers as independent graph segments."""

import logging
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any, Optional, Union

import torch
from torch import nn

from hyper_parallel.distributed.npu_inductor_shim import (  # isort: skip
    apply_npu_kernel_fallbacks,
    fallback_spec,
    install_npu_flattened_dims_shim,
    install_rope_eager_break,
    rope_eager_is_requested,
    shim_is_requested,
)
from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager

logger = logging.getLogger(__name__)


_MAPPING_GET_POLYFILL_INSTALLED = False

# No per-model layer-path table: the model-owned ``get_compile_layers()``
# contract comes first, and the generic HF-convention fallback paths below
# cover every supported family. A family whose container lives elsewhere
# declares the contract on its model class instead of registering here.
#
# VLM families keep the text decoder under a tower attribute, and those paths
# are tried first for two reasons: a vision tower exposes ``layers`` of its own
# (Kimi-K2.5 has 27 of them) while the text decoder is the part worth compiling,
# and a VLM whose top-level object is a thin wrapper adds one ``model.`` level.
# A pure text model simply misses every tower path and falls through.
_GENERIC_LAYER_PATHS = (
    "model.model.language_model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "language_model.layers",
    "layers",
    "model.layers",
)


def _get_attribute(root: Any, path: str) -> Any:
    """Read a dotted module path, including numeric ModuleList indexes."""
    value = root
    for part in path.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return value


def _is_named_layer(item: Any) -> bool:
    """Return whether an item is a valid ``(fqn, module)`` pair."""
    if not isinstance(item, tuple) or len(item) != 2:
        return False
    name, layer = item
    return isinstance(name, str) and isinstance(layer, nn.Module)


def _normalize_declared_layers(declared: Any) -> list[tuple[str, nn.Module]]:
    """Normalize the model-owned decoder-layer contract."""
    if isinstance(declared, nn.ModuleList):
        return [(str(index), layer) for index, layer in enumerate(declared)]
    if not isinstance(declared, Iterable) or isinstance(declared, (str, bytes)):
        raise TypeError(
            "get_compile_layers() must return an iterable of modules or "
            "(fqn, module) pairs"
        )

    layers = []
    for index, item in enumerate(declared):
        if isinstance(item, nn.Module):
            layers.append((str(index), item))
            continue
        if _is_named_layer(item):
            layers.append(item)
            continue
        raise TypeError(
            "get_compile_layers() entries must be modules or (fqn, module) pairs, "
            f"but entry {index} is {type(item).__name__}"
        )
    return layers


def _qualify_declared_layers(
    model: nn.Module,
    layers: list[tuple[str, nn.Module]],
) -> list[tuple[str, nn.Module]]:
    """Replace generated numeric names with the modules' model-owned FQNs."""
    module_names = {id(module): name for name, module in model.named_modules()}
    return [
        (module_names.get(id(layer), name) if name.isdigit() else name, layer)
        for name, layer in layers
    ]


def _layers_from_path(model: nn.Module, path: str) -> list[tuple[str, nn.Module]]:
    """Return indexed layers from one declared container path."""
    try:
        container = _get_attribute(model, path)
    except (AttributeError, IndexError, KeyError, TypeError):
        return []
    if not isinstance(container, (nn.ModuleList, nn.Sequential)):
        return []
    return [(f"{path}.{index}", layer) for index, layer in enumerate(container)]


def get_compile_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return stable decoder-layer segments declared by a supported model.

    Resolution order: the model-owned ``get_compile_layers()`` contract
    first, then the generic HF-convention container paths
    (``_GENERIC_LAYER_PATHS``). A family whose container follows neither
    declares ``get_compile_layers()`` on its model class.
    """
    declared_getter = getattr(model, "get_compile_layers", None)
    if callable(declared_getter):
        layers = _qualify_declared_layers(
            model,
            _normalize_declared_layers(declared_getter()),
        )
        if layers:
            return layers

    seen_containers = set()
    for path in _GENERIC_LAYER_PATHS:
        layers = _layers_from_path(model, path)
        if not layers:
            continue
        container_id = id(_get_attribute(model, path))
        if container_id in seen_containers:
            continue
        seen_containers.add(container_id)
        return layers

    raise ValueError(
        "compile is enabled, but the model exposes no decoder-layer compile contract; "
        "define get_compile_layers() or use a supported model layer container"
    )


def _install_dynamo_mapping_get_polyfill() -> None:
    """Make ``Mapping.get`` traceable without changing Transformers source.

    Transformers' attention registry calls ``Mapping.get`` from every decoder
    layer. TorchDynamo treats the standard-library implementation as a
    skipfile, so the call creates a graph break. ``substitute_in_graph`` only
    replaces the implementation while Dynamo inlines it; eager execution
    continues to use the original method.
    """
    # Module-level idempotency flag: the polyfill must be installed at most
    # once per process, so a global statement is required here.
    global _MAPPING_GET_POLYFILL_INSTALLED  # pylint: disable=global-statement
    if _MAPPING_GET_POLYFILL_INSTALLED:
        return

    substitute_in_graph = getattr(torch.compiler, "substitute_in_graph", None)
    if substitute_in_graph is None:
        logger.warning(
            "Torch does not provide compiler.substitute_in_graph; "
            "Mapping.get graph breaks cannot be removed"
        )
        _MAPPING_GET_POLYFILL_INSTALLED = True
        return

    def _mapping_get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    try:
        substitute_in_graph(Mapping.get)(_mapping_get)
    except ValueError as exc:
        if "Duplicate dispatch rule" not in str(exc):
            raise
        logger.debug("Mapping.get already has a TorchDynamo substitution")
    _MAPPING_GET_POLYFILL_INSTALLED = True


def resolve_compile_kwargs(config: CompileConfig) -> dict[str, Any]:
    """Convert ``CompileConfig`` into keyword arguments for ``Module.compile``."""
    kwargs: dict[str, Any] = {
        "fullgraph": config.fullgraph,
        "dynamic": config.dynamic,
    }
    if config.backend is not None:
        kwargs["backend"] = config.backend
    if config.options:
        kwargs["options"] = dict(config.options)
    else:
        kwargs["mode"] = config.mode
    return kwargs


def _resolve_compile_targets(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return the modules to compile, optionally inside the checkpoint wrappers.

    Compiling the decoder layers while activation checkpointing is on puts the
    checkpoint context manager *inside* the traced region, and dynamo refuses to
    enter a ``_GeneratorContextManager``: every layer then breaks its graph at the
    entry and silently runs eagerly, so nothing is ever fused.  Measured on the
    2-SN shape: HP logged "Compiled 61 decoder layers", the cache dir stayed empty,
    and the device kernel inventory of a compile-enabled arm was byte-identical to
    the uncompiled one (34856 kernels / 144 distinct, 0 only-in-either).

    ``HP_COMPILE_INSIDE_AC=1`` compiles the module held *by* each checkpoint wrapper
    instead, which leaves the context manager outside the compiled graph while still
    covering the math (attention projections, MoE routing/permute, elementwise) that
    the compiler is supposed to fuse.  Default off: unchanged behaviour.

    Args:
        model: Model whose compile targets should be resolved.

    Returns:
        ``(fqn, module)`` pairs to compile.
    """
    if os.environ.get("HP_COMPILE_INSIDE_AC", "0") != "1":
        return get_compile_layers(model)

    from hyper_parallel.distributed.activation_checkpoint import (  # pylint: disable=import-outside-toplevel
        _find_checkpoint_wrappers,
        _get_checkpoint_wrapped_module,
    )

    wrappers = _find_checkpoint_wrappers(model)
    targets = []
    for path, wrapper in wrappers.items():
        inner = _get_checkpoint_wrapped_module(wrapper)
        if isinstance(inner, nn.Module):
            targets.append((path, inner))

    # ``HP_COMPILE_INSIDE_AC_INCLUDE`` narrows the targets further by FQN regex.  It
    # exists because compiling a MoE block makes execution fail inside the fused
    # grouped GEMM (`npu_grouped_matmul ... call aclnnGroupedMatmulV5 failed, error
    # code is 1`), while the attention blocks are where most of the copy/reshape glue
    # lives (MLA rebuilds q/k/v with transpose/strided-slice/cat).  Empty (default)
    # keeps every wrapped block.
    include = os.environ.get("HP_COMPILE_INSIDE_AC_INCLUDE", "").strip()
    if include:
        try:
            pattern = re.compile(include)
        except re.error as error:
            logger.warning(
                "HP_COMPILE_INSIDE_AC_INCLUDE=%r is not a valid regular expression (%s); "
                "compiling every wrapped block instead.",
                include,
                error,
            )
        else:
            narrowed = [(path, module) for path, module in targets if pattern.search(path)]
            flat = os.environ.get("HP_COMPILE_DESCENDANTS", "0") == "1"
            if not narrowed and flat:
                # A wrapped block can be a whole transformer layer, whose FQN carries no
                # "attn" even though it *contains* the attention modules.  Matching the
                # block FQNs only is how the first 2-SN compile arms ended up compiling the
                # vision tower instead of the text tower: the config lists exactly 108
                # vision-tower attention leaf specs, and "attn" selected those 108 -- a
                # frozen subtree worth <1% of the step, whose modules are single matmuls
                # with nothing to fuse.  With HP_COMPILE_DESCENDANTS=1 the pattern is
                # matched against each wrapped block's descendants and the matching child
                # is compiled instead, so `language_model.*self_attn$` reaches the text
                # attention without pulling in the MoE grouped GEMM that crashes.
                narrowed = _select_descendants(targets, pattern)
            logger.info(
                "Compile target filter %r kept %d wrapped block(s) as %d compile target(s)%s",
                include, len([p for p, _ in targets if pattern.search(p)]) or len(targets),
                len(narrowed), " (descendant match)" if flat else "",
            )
            if narrowed:
                _log_target_preview(include, narrowed)
            targets = narrowed

    return targets or get_compile_layers(model)


def _select_descendants(
    targets: list[tuple[str, nn.Module]],
    pattern: "re.Pattern[str]",
) -> list[tuple[str, nn.Module]]:
    """Return the deepest descendants of ``targets`` whose FQN matches ``pattern``.

    Deepest-match keeps the compiled region as small as the pattern asks for: matching
    ``self_attn$`` yields the attention module itself rather than each projection inside
    it, which is where the copy/reshape glue of an MLA rebuild lives.

    Args:
        targets: ``(fqn, module)`` pairs of the wrapped blocks to search.
        pattern: compiled FQN regular expression.

    Returns:
        ``(fqn, module)`` pairs, deduplicated by module identity.
    """
    seen: set[int] = set()
    matches: list[tuple[str, nn.Module]] = []
    for path, module in targets:
        for rel, child in module.named_modules():
            fqn = f"{path}.{rel}" if rel else path
            if not pattern.search(fqn) or id(child) in seen:
                continue
            seen.add(id(child))
            matches.append((fqn, child))
    # Keep only the outermost matches: compiling a parent already covers its children, and
    # nested ``torch.compile`` wrappers are exactly the failure mode this module exists to
    # avoid.  ``self_attn$`` therefore yields the attention module, not each projection.
    return [
        (fqn, child)
        for fqn, child in matches
        if not any(fqn.startswith(other + ".") for other, _ in matches)
    ]


def _log_target_preview(include: str, targets: list[tuple[str, nn.Module]]) -> None:
    """Log which subtrees a compile filter selected, so a mis-target is visible."""
    families: dict[str, int] = {}
    for fqn, _ in targets:
        head = ".".join(fqn.split(".")[:3])
        families[head] = families.get(head, 0) + 1
    logger.info("  filter %r target families: %s", include, dict(sorted(families.items())))
    logger.info("  filter %r first targets: %s", include, [fqn for fqn, _ in targets[:4]])


def apply_compile(model: nn.Module, config: CompileConfig) -> nn.Module:
    """Compile each decoder layer in place while preserving module identity."""
    if not config.enabled:
        return model

    _install_dynamo_mapping_get_polyfill()
    if shim_is_requested():
        install_npu_flattened_dims_shim()
        apply_npu_kernel_fallbacks(fallback_spec())
    if rope_eager_is_requested():
        install_rope_eager_break()
    torch._dynamo.config.cache_size_limit = config.dynamo_cache_size_limit  # pylint: disable=W0212
    compile_kwargs = resolve_compile_kwargs(config)
    layers = _resolve_compile_targets(model)
    for layer_fqn, layer in layers:
        try:
            layer.compile(**compile_kwargs)
        except Exception as exc:
            raise RuntimeError(f"failed to compile segment {layer_fqn}") from exc

    # Report the backend that will actually run: ``resolve_compile_kwargs`` omits the
    # argument when it is unset, so the effective value is torch.compile's default
    # ("inductor"), not the ``None`` that ``config.backend`` holds.  Logging the raw config
    # value sent me chasing a nonexistent "backend=None" no-op for a while.
    logger.info(
        "Compiled %d decoder layers (backend=%s, mode=%s, dynamic=%s, fullgraph=%s)",
        len(layers),
        compile_kwargs.get("backend", "inductor"),
        config.mode,
        config.dynamic,
        config.fullgraph,
    )
    return model


__all__ = [
    "apply_compile",
    "get_compile_layers",
    "resolve_compile_kwargs",
]


def _resolve_compile_config(
    compile_config: Optional[Union[CompileConfig, dict]],
    validate_placement: bool,
    fsdp2_manager: Optional[FSDP2Manager],
) -> tuple[Optional[CompileConfig], bool]:
    """Normalize compile configuration and validate its FSDP interaction.

    Moved from ``_transformers/infrastructure.py`` (05 §15.2.1): compile
    option normalization and the fullgraph/FSDP conflict check follow the
    compile feature.
    """
    if isinstance(compile_config, dict):
        compile_config = CompileConfig(enabled=True, **compile_config)
    if compile_config is not None and not isinstance(compile_config, CompileConfig):
        raise TypeError("compile_config must be a CompileConfig, mapping, or None")
    compile_for_execution = bool(
        not validate_placement and compile_config is not None and compile_config.enabled
    )
    if validate_placement and compile_config is not None and compile_config.enabled:
        logger.info("Skipping decoder-layer compile during placement validation")
    if compile_for_execution and compile_config.fullgraph and isinstance(fsdp2_manager, FSDP2Manager):
        raise ValueError(
            "compile.fullgraph=True is incompatible with FSDP hooks kept eager "
            "by _dynamo_disable; set compile.fullgraph=False"
        )
    return compile_config, compile_for_execution
