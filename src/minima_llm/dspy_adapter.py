# dspy_adapter.py
"""DSPy integration for MinimaLlm."""
from __future__ import annotations

import typing
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

import asyncio
import contextvars
import inspect
import os
import re


try:
    import dspy
    from dspy.adapters.chat_adapter import ChatAdapter
except ImportError as e:
    raise ImportError(
        "This module requires DSPy. Install with: pip install minima-llm[dspy]"
    ) from e

def _import_adapter_parse_error():
    """Locate AdapterParseError across DSPy versions."""
    paths = [
        "dspy.adapters.exceptions",
        "dspy.adapters.base",
        "dspy.adapters",
        "dspy.primitives.exceptions",
        "dspy.exceptions",
        "dspy.utils.exceptions",
    ]

    for path in paths:
        try:
            module = __import__(path, fromlist=["AdapterParseError"])
            return module.AdapterParseError
        except (ImportError, AttributeError):
            continue

    # Fallback: define compatible exception for older DSPy versions
    dspy_version = getattr(dspy, "__version__", "unknown")
    print(f"Warning: AdapterParseError not found in DSPy {dspy_version}, using fallback class")

    class AdapterParseError(Exception):
        """Fallback AdapterParseError for DSPy versions without this exception."""
        def __init__(self, adapter_name=None, signature=None, lm_response=None,
                     parsed_result=None, message=None, **kwargs):
            self.adapter_name = adapter_name
            self.signature = signature
            self.lm_response = lm_response
            self.parsed_result = parsed_result
            super().__init__(message or "Adapter parse error")

    return AdapterParseError


AdapterParseError = _import_adapter_parse_error()

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore

from .protocol import MinimaLlmRequest
from .backend import (
    MinimaLlmFailure,
    OpenAIMinimaLlm,
    get_force_refresh,
    reset_force_refresh,
    set_force_refresh,
    set_last_cached,
)
from .config import MinimaLlmConfig


# ====== More tolerant chat adapter ========


class TolerantChatAdapter(ChatAdapter):
    # Matches a well-formed header anywhere in a line, e.g. [[ ## answerability ## ]]
    _HEADER_RE = re.compile(
        r"\[\[\s*##\s*(?P<name>[^#\]\r\n]+?)\s*##\s*\]\]",
        flags=re.IGNORECASE,
    )

    @classmethod
    def normalize_field_name(cls, raw: str) -> str:
        return raw.strip().lower().replace(" ", "_")

    @classmethod
    def is_optional_type(cls, tp):
        """Return True if annotation is Optional[...]."""
        return (
            getattr(tp, "__origin__", None) is typing.Union
            and type(None) in getattr(tp, "__args__", ())
        )

    @classmethod
    def unwrap_optional(cls, ann) -> type:
        """Unwrap Optional[T] to T, or return ann unchanged."""
        if getattr(ann, "__origin__", None) is typing.Union:
            args = getattr(ann, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return non_none[0]
        return ann

    @classmethod
    def is_type(cls, ann, target: type) -> bool:
        """Return True if annotation is target or Optional[target]."""
        return cls.unwrap_optional(ann) is target

    # Convenience methods for common types
    is_float = classmethod(lambda cls, ann: cls.is_type(ann, float))
    is_int = classmethod(lambda cls, ann: cls.is_type(ann, int))

    # Regex patterns for numeric parsing
    _FLOAT_PATTERN = re.compile(r"[-+]?[0-9]*\.?[0-9]+")
    _INT_PATTERN = re.compile(r"[-+]?\d+")

    @classmethod
    def try_parse_numeric(cls, val, target: type, pattern: re.Pattern) -> int | float:
        """Parse numeric value from LLM output. Raises ValueError on failure."""
        try:
            m = pattern.search(str(val).strip())
            if m:
                return target(m.group())
        except (ValueError, TypeError):
            pass
        raise ValueError(f"Could not parse {target.__name__}: {str(val)[:50]}")

    @classmethod
    def is_list_str(cls, ann):
        """Return True if annotation is list[str], List[str], or Optional[list[str]]."""
        from typing import Union
        origin = getattr(ann, "__origin__", None)

        # Handle Optional[List[str]] -> Union[List[str], None]
        if origin is Union:
            args = getattr(ann, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return cls.is_list_str(non_none[0])
            return False

        if origin is list:
            args = getattr(ann, "__args__", ())
            return args == (str,)
        return False

    # Regex patterns for list format detection
    _SINGLE_QUOTE_LIST = re.compile(r"^\s*\[\s*'")
    _DOUBLE_QUOTE_LIST = re.compile(r'^\s*\[\s*"')
    _UNESCAPED_SINGLE = re.compile(r"(?<!\\)'")

    @classmethod
    def try_parse_list_str(cls, val: str) -> list[str]:
        """Parse list from JSON or Python syntax."""
        import ast
        import json

        val = val.strip()

        if cls._DOUBLE_QUOTE_LIST.match(val):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x]
            except json.JSONDecodeError:
                pass
            raise ValueError(f"Invalid JSON array: {val[:100]}")

        elif cls._SINGLE_QUOTE_LIST.match(val):
            quote_count = len(cls._UNESCAPED_SINGLE.findall(val))
            if quote_count % 2 != 0:
                raise ValueError(
                    f"Unbalanced single quotes ({quote_count}) - "
                    f"likely apostrophe issue, use JSON format: {val[:100]}"
                )

            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x]
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Invalid Python list ({e}), use JSON format: {val[:100]}")

        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except json.JSONDecodeError:
            pass

        raise ValueError(f"Expected JSON array, got: {val[:100]}")

    @classmethod
    def _is_non_value(cls, s: str) -> bool:
        return s.strip().lower() in {"", "none", "null"}

    def _extract_sections(self, completion: str) -> list[tuple[str | None, list[str]]]:
        """Extract (header, lines) sections from completion."""
        sections: list[tuple[str | None, list[str]]] = [(None, [])]
        current_lines = sections[-1][1]

        for raw_line in completion.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            last_end = 0
            for m in self._HEADER_RE.finditer(line):
                before = line[last_end:m.start()].strip()
                if before:
                    current_lines.append(before)

                header = self.normalize_field_name(m.group("name"))
                sections.append((header, []))
                current_lines = sections[-1][1]
                last_end = m.end()

            after = line[last_end:].strip()
            if after:
                current_lines.append(after)

        return sections

    def _sections_to_dict(self, sections: list[tuple[str | None, list[str]]], output_fields: set[str]) -> dict[str, str]:
        """Reduce sections to {field: value} for known output fields."""
        parsed: dict[str, str] = {}
        for key, lines in sections:
            if key in output_fields:
                val = "\n".join(lines).strip()
                if not self._is_non_value(val):
                    parsed[key] = val
        return parsed

    def _coerce_field(self, val: str, annotation: type) -> typing.Any:
        """Coerce string value to annotation type."""
        if self.is_float(annotation):
            return float(self.try_parse_numeric(val, float, self._FLOAT_PATTERN))
        if self.is_int(annotation):
            return int(self.try_parse_numeric(val, int, self._INT_PATTERN))
        if self.is_list_str(annotation) and isinstance(val, str):
            return list(self.try_parse_list_str(val))
        return val

    def _validate_and_coerce(self, parsed: dict[str, str], signature, completion: str) -> dict[str, typing.Any]:
        """Validate required fields present and coerce all values."""
        result: dict[str, typing.Any] = {}

        for name, field in signature.output_fields.items():
            annotation = field.annotation

            if name in parsed:
                try:
                    result[name] = self._coerce_field(parsed[name], annotation)
                except (ValueError, TypeError) as e:
                    raise AdapterParseError(
                        adapter_name="TolerantChatAdapter",
                        signature=signature,
                        lm_response=completion,
                        parsed_result=parsed,
                        message=str(e),
                    )
            elif self.is_optional_type(annotation):
                result[name] = None
            else:
                raise AdapterParseError(
                    adapter_name="TolerantChatAdapter",
                    signature=signature,
                    lm_response=completion,
                    parsed_result=parsed,
                    message=f"Missing required field: {name}",
                )

        return result

    def parse(self, signature, completion: str) -> dict[str, typing.Any]:
        """Parse LLM completion into typed output fields."""
        sections = self._extract_sections(completion)
        parsed = self._sections_to_dict(sections, set(signature.output_fields.keys()))
        return self._validate_and_coerce(parsed, signature, completion)


def _get_dspy_version() -> tuple[int, int, int]:
    """Parse DSPy version into (major, minor, patch) tuple."""
    version_str = getattr(dspy, "__version__", "0.0.0")
    try:
        parts = version_str.split(".")[:3]
        return tuple(int(p) for p in parts)  # type: ignore
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _select_adapter():
    """Select adapter for DSPy."""
    return TolerantChatAdapter()


_dspy_version = _get_dspy_version()
_adapter = _select_adapter()
print(f"DSPy {'.'.join(map(str, _dspy_version))} loaded, using {type(_adapter).__name__}")
dspy.settings.configure(adapter=_adapter)



# ==============

def _resolve_dspy_base_lm() -> Type[Any]:
    """Locate DSPy's BaseLM class across common DSPy layouts."""
    if hasattr(dspy, "BaseLM"):
        return dspy.BaseLM  # type: ignore[attr-defined]

    for mod_name, attr in [
        ("dspy.clients", "BaseLM"),
        ("dspy.clients.base", "BaseLM"),
        ("dspy.clients.lm", "BaseLM"),
        ("dspy.models", "BaseLM"),
        ("dspy.lm", "BaseLM"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            if hasattr(mod, attr):
                return getattr(mod, attr)
        except Exception:
            pass

    raise RuntimeError("Could not locate DSPy BaseLM")


_BaseLM = _resolve_dspy_base_lm()


class MinimaLlmDSPyLM(_BaseLM):  # type: ignore[misc]
    """
    DSPy BaseLM adapter that routes calls through OpenAIMinimaLlm.

    This adapter is intentionally minimal:
      - DSPy handles prompt construction and output parsing.
      - MinimaLlm handles HTTP transport, backpressure, retries, and pacing.
      - No LiteLLM dependency.
    """

    def __init__(self, minimallm: OpenAIMinimaLlm, **kwargs: Any):
        self._minimallm = minimallm
        model_value = minimallm.cfg.model

        try:
            sig = inspect.signature(_BaseLM.__init__)  # type: ignore[arg-type]
            params = sig.parameters
            init_kwargs: Dict[str, Any] = {}

            if "model" in params:
                init_kwargs["model"] = model_value
            elif "model_name" in params:
                init_kwargs["model_name"] = model_value

            for k, v in kwargs.items():
                if k in params:
                    init_kwargs[k] = v

            super().__init__(**init_kwargs)  # type: ignore[misc]
        except Exception:
            try:
                super().__init__(model=model_value)  # type: ignore[misc]
            except Exception:
                try:
                    super().__init__(model_value)  # type: ignore[misc]
                except Exception:
                    super().__init__()  # type: ignore[misc]

        if not hasattr(self, "model"):
            self.model = model_value  # type: ignore[assignment]
        if not hasattr(self, "kwargs"):
            self.kwargs = {}  # type: ignore[assignment]
        if not hasattr(self, "history"):
            self.history = []  # type: ignore[assignment]

    async def acall(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """Async LM call used by DSPy."""
        force_refresh = force_refresh or get_force_refresh()

        if messages is None:
            if prompt is None:
                raise ValueError("DSPy LM requires either prompt or messages")
            messages = [{"role": "user", "content": prompt}]

        if os.environ.get("MINIMA_DEBUG") and kwargs:
            print(f"[MinimaLlmDSPyLM] DSPy kwargs: {list(kwargs.keys())}")
            if "response_format" in kwargs:
                print(f"[MinimaLlmDSPyLM] response_format: {kwargs['response_format']}")

        req = MinimaLlmRequest(
            request_id=str(kwargs.pop("request_id", "dspy")),
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=kwargs.pop("temperature", None),
            max_tokens=kwargs.pop("max_tokens", None),
            extra=kwargs if kwargs else None,
        )

        resp = await self._minimallm.generate(req, force_refresh=force_refresh)
        if isinstance(resp, MinimaLlmFailure):
            error_msg = f"{resp.error_type}: {resp.message}"
            if resp.body_snippet:
                error_msg += f"\nResponse body: {resp.body_snippet}"
            raise RuntimeError(error_msg)
        set_last_cached(resp.cached)
        return [resp.text]

    async def aforward(self, *args: Any, **kwargs: Any) -> List[str]:
        return await self.acall(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> List[str]:
        return self.__call__(*args, **kwargs)

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Sync LM call fallback."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "MinimaLlmDSPyLM was called synchronously inside a running event loop. "
                "Use await pred.acall(...) or await lm.acall(...)."
            )

        return asyncio.run(self.acall(prompt=prompt, messages=messages, **kwargs))


# ----------------------------
# Batch execution helper
# ----------------------------

def _get_input_field_names(signature_class: Type[dspy.Signature]) -> List[str]:
    """Extract InputField names from a DSPy Signature class."""
    input_fields = []

    for attr_name in ['input_fields', '_input_fields', 'fields']:
        if hasattr(signature_class, attr_name):
            fields_obj = getattr(signature_class, attr_name)

            if isinstance(fields_obj, dict):
                for key, value in fields_obj.items():
                    if isinstance(key, str):
                        input_fields.append(key)
                if input_fields:
                    break
            elif hasattr(fields_obj, 'keys') and callable(fields_obj.keys):
                try:
                    input_fields = list(fields_obj.keys())
                    if input_fields:
                        break
                except Exception:
                    pass
            elif hasattr(fields_obj, '__iter__'):
                try:
                    input_fields = [f for f in fields_obj if isinstance(f, str)]
                    if input_fields:
                        break
                except Exception:
                    pass

    if input_fields:
        return input_fields

    if hasattr(signature_class, 'model_fields'):
        fields_dict = signature_class.model_fields
        for name, field_info in fields_dict.items():
            if hasattr(field_info, 'metadata') and field_info.metadata:
                for meta_item in field_info.metadata:
                    meta_type = type(meta_item).__name__
                    meta_module = type(meta_item).__module__ if hasattr(type(meta_item), '__module__') else ''
                    if 'InputField' in meta_type or ('dspy' in meta_module and 'Input' in meta_type):
                        input_fields.append(name)
                        break

            if name not in input_fields and hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                extra = field_info.json_schema_extra
                if extra.get('__dspy_field_type') == 'input':
                    input_fields.append(name)
                elif extra.get('prefix', '').lower().startswith('input'):
                    input_fields.append(name)

    if input_fields:
        return input_fields

    if hasattr(signature_class, '__fields__'):
        fields_dict = signature_class.__fields__
        for name, field_info in fields_dict.items():
            if hasattr(field_info, 'field_info'):
                field_info = field_info.field_info
            if hasattr(field_info, 'extra') and field_info.extra:
                extra = field_info.extra
                if extra.get('__dspy_field_type') == 'input':
                    input_fields.append(name)
            if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                extra = field_info.json_schema_extra
                if extra.get('__dspy_field_type') == 'input':
                    input_fields.append(name)

    if input_fields:
        return input_fields

    for name in signature_class.__annotations__:
        try:
            field_obj = getattr(signature_class, name, None)
            if field_obj is None:
                field_obj = signature_class.__dict__.get(name, None)

            if field_obj is None:
                continue

            field_type_str = str(type(field_obj))
            field_class_name = field_obj.__class__.__name__ if hasattr(field_obj, '__class__') else ''
            field_module = field_obj.__class__.__module__ if hasattr(field_obj, '__class__') else ''

            is_input = any([
                'InputField' in field_class_name,
                'InputField' in field_type_str,
                'Input' in field_class_name and 'dspy' in field_module,
                hasattr(field_obj, 'json_schema_extra') and
                    isinstance(field_obj.json_schema_extra, dict) and
                    field_obj.json_schema_extra.get('__dspy_field_type') == 'input',
            ])

            if is_input:
                input_fields.append(name)

        except Exception:
            continue

    if input_fields:
        return input_fields

    print(f"Warning: Could not detect InputFields via metadata, using annotation order heuristic")

    output_field_names = []

    for name in signature_class.__annotations__:
        try:
            field_obj = getattr(signature_class, name, None)
            if field_obj is None:
                field_obj = signature_class.__dict__.get(name, None)

            if field_obj is not None:
                field_type_str = str(type(field_obj))
                field_class_name = field_obj.__class__.__name__ if hasattr(field_obj, '__class__') else ''

                if 'OutputField' in field_class_name or 'OutputField' in field_type_str:
                    output_field_names.append(name)
        except Exception:
            continue

    if output_field_names:
        first_output = output_field_names[0]
        for name in signature_class.__annotations__:
            if name == first_output:
                break
            input_fields.append(name)
        return input_fields

    annotations = list(signature_class.__annotations__.keys())
    if len(annotations) >= 4:
        print(f"Warning: Using first 4 annotations as inputs: {annotations[:4]}")
        return annotations[:4]

    print(f"Error: Could not detect any InputFields in signature {signature_class.__name__}")
    return []


async def run_dspy_batch(
    signature_class: Type[dspy.Signature],
    annotation_objs: List["BaseModel"],
    output_converter: Callable[[Any, "BaseModel"], None],
    backend: OpenAIMinimaLlm,
    predictor_class: Type = dspy.ChainOfThought,
) -> List["BaseModel"]:
    """
    Execute a DSPy batch with MinimaLLM backend.

    Parameters
    ----------
    signature_class : Type[dspy.Signature]
        DSPy Signature class (e.g., Umbrela)
    annotation_objs : List[BaseModel]
        List of Pydantic models with fields matching signature InputFields
    output_converter : Callable[[Any, BaseModel], None]
        Function that updates annotation object with DSPy prediction result.
    predictor_class : Type
        DSPy predictor class (default: dspy.ChainOfThought)
    backend : OpenAIMinimaLlm
        Pre-configured backend.

    Returns
    -------
    List[BaseModel]
        Processed annotation objects with outputs filled in
    """
    owns_backend = backend is None
    if backend is None:
        backend = OpenAIMinimaLlm.from_env()

    lm = MinimaLlmDSPyLM(backend)

    input_fields = _get_input_field_names(signature_class)

    CODE_ERRORS = (NameError, TypeError, AttributeError, SyntaxError, ImportError)

    http_max_attempts = backend.cfg.max_attempts
    parse_retry_limit = 3 if http_max_attempts == 0 else http_max_attempts

    with dspy.context(lm=lm, adapter=_select_adapter()):
        predictor = predictor_class(signature_class)

        async def _maybe_await(result):
            if inspect.isawaitable(result):
                return await result
            return result

        async def invoke_predictor(pred, **kw):
            import functools

            for method_name in ("acall", "aforward"):
                method = getattr(pred, method_name, None)
                if callable(method):
                    return await _maybe_await(method(**kw))

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(pred, **kw))

        async def process_one(obj: "BaseModel") -> "BaseModel":
            kw = obj.model_dump(include=set(input_fields))
            last_error: Optional[Exception] = None

            for attempt in range(parse_retry_limit):
                force_refresh_token: Optional[contextvars.Token[bool]] = None
                try:
                    if attempt > 0:
                        force_refresh_token = set_force_refresh(True)

                    result = await invoke_predictor(predictor, **kw)
                    output_converter(result, obj)
                    return obj

                except CODE_ERRORS:
                    raise
                except AdapterParseError as e:
                    last_error = e
                    continue
                except Exception as e:
                    last_error = e
                    continue
                finally:
                    if force_refresh_token is not None:
                        reset_force_refresh(force_refresh_token)

            raise last_error  # type: ignore[misc]

        results = await backend.run_batched_callable(annotation_objs, process_one)

    failures = [r for r in results if isinstance(r, MinimaLlmFailure)]
    if failures:
        if owns_backend:
            await backend.aclose()
        msgs = [f"{f.request_id}: {f.error_type}: {f.message}" for f in failures[:5]]
        raise RuntimeError(
            f"{len(failures)} DSPy predictions failed:\n  " + "\n  ".join(msgs)
        )

    if owns_backend:
        await backend.aclose()

    return cast(List["BaseModel"], results)


async def _collect_requests_for_batch(
    signature_class: Type[dspy.Signature],
    annotation_objs: List["BaseModel"],
    backend: "OpenAIMinimaLlm",
    predictor_class: Type = None,
) -> None:
    """Phase 1 of batch execution: Run through DSPy to queue all requests."""
    if predictor_class is None:
        predictor_class = dspy.ChainOfThought

    lm = MinimaLlmDSPyLM(backend)
    input_fields = _get_input_field_names(signature_class)

    with dspy.context(lm=lm, adapter=_select_adapter()):
        predictor = predictor_class(signature_class)

        async def _maybe_await(result):
            if inspect.isawaitable(result):
                return await result
            return result

        async def invoke_predictor(pred, **kw):
            import functools

            for method_name in ("acall", "aforward"):
                method = getattr(pred, method_name, None)
                if callable(method):
                    return await _maybe_await(method(**kw))

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(pred, **kw))

        async def collect_one(obj: "BaseModel") -> "BaseModel":
            kw = obj.model_dump(include=set(input_fields))
            try:
                await invoke_predictor(predictor, **kw)
            except Exception:
                pass
            return obj

        await backend.run_batched_callable(annotation_objs, collect_one)


T = typing.TypeVar("T", bound="BaseModel")


def run_dspy_batch_generic(
    data: List[T],
    signature: Type[dspy.Signature],
    converter: Callable[[dspy.Prediction, T], None],
    llm_config: "MinimaLlmConfig",
) -> List[T]:
    """
    Run DSPy batch for any data model and signature.

    Convenience wrapper around run_dspy_batch that handles asyncio
    and backend setup. Supports Parasail batch mode for 50% cost savings.
    """
    if not data:
        return data

    backend = OpenAIMinimaLlm(llm_config)

    if llm_config.parasail.prefix:
        async def _run_batch_mode():
            async with backend.batch_mode(llm_config.parasail.prefix):
                await _collect_requests_for_batch(signature, data, backend)

            return await run_dspy_batch(signature, data, converter, backend=backend)

        return asyncio.run(_run_batch_mode())

    return asyncio.run(
        run_dspy_batch(
            signature,
            data,
            converter,
            backend=backend,
        )
    )


def print_dspy_prompt(sig: dspy.Signature, inputs: Dict[str, Any]):
    predict = dspy.Predict(sig)

    adapter = dspy.settings.adapter

    messages = adapter.format(
        signature=predict.signature,
        demos=[],
        inputs=inputs
    )

    print(messages)
