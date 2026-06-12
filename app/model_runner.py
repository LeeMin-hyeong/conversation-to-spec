from __future__ import annotations

import random
import os
import sys
import tempfile
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any


class BaseModelRunner(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.last_generation_info: dict[str, Any] = {}

    @abstractmethod
    def generate(self, prompt: str, generation_config: dict) -> str:
        raise NotImplementedError

    def prepare(self) -> None:
        return None

    def close(self) -> None:
        return None


class MLXModelRunner(BaseModelRunner):
    """Local MLX runner for Apple Silicon text-generation models."""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name=model_name)
        self._mx = None
        self._model = None
        self._tokenizer = None
        self._generate = None
        self._make_sampler = None
        self._last_generation_args: dict[str, Any] = {}

    @staticmethod
    @contextmanager
    def _filter_mlx_stderr():
        stderr_fd = sys.stderr.fileno()
        saved_fd = os.dup(stderr_fd)
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            try:
                os.dup2(tmp.fileno(), stderr_fd)
                yield
            finally:
                os.dup2(saved_fd, stderr_fd)
                os.close(saved_fd)
                tmp.seek(0)
                captured = tmp.read().decode(errors="replace")
                for line in captured.splitlines():
                    if "mx.metal.device_info is deprecated" in line:
                        continue
                    print(line, file=sys.stderr)

    def _ensure_loaded(self) -> None:
        if (
            self._model is not None
            and self._tokenizer is not None
            and self._generate is not None
            and self._make_sampler is not None
        ):
            return
        try:
            import mlx.core as mx  # type: ignore
            from mlx_lm import generate, load  # type: ignore
            from mlx_lm.sample_utils import make_sampler  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional runtime env
            raise RuntimeError("mlx-lm is required for MLXModelRunner.") from exc
        self._mx = mx
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*mx\.metal\.device_info is deprecated.*",
            )
            self._model, self._tokenizer = load(self.model_name)
        self._generate = generate
        self._make_sampler = make_sampler

    def prepare(self) -> None:
        self._ensure_loaded()

    def _prompt_text(self, prompt: str) -> str:
        assert self._tokenizer is not None
        tokenizer = getattr(self._tokenizer, "tokenizer", self._tokenizer)
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if apply_chat_template is None:
            return prompt
        try:
            return apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            try:
                return apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return prompt

    def _prompt_token_count(self, prompt_text: str) -> int | None:
        assert self._tokenizer is not None
        try:
            return len(self._tokenizer.encode(prompt_text))
        except Exception:
            return None

    def _generation_args(self, generation_config: dict) -> dict[str, Any]:
        assert self._make_sampler is not None
        temperature = float(generation_config.get("temperature", 0.0))
        do_sample = bool(generation_config.get("do_sample", False))
        if temperature <= 0 or not do_sample:
            temperature = 0.0
            top_p = 0.0
        else:
            top_p = float(generation_config.get("top_p", 1.0))
        args: dict[str, Any] = {
            "max_tokens": int(generation_config.get("max_new_tokens", 900)),
            "sampler": self._make_sampler(temp=temperature, top_p=top_p),
        }
        self._last_generation_args = {
            "max_tokens": args["max_tokens"],
            "do_sample": do_sample and temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
        }
        return args

    def _apply_seed(self, generation_config: dict) -> None:
        if self._mx is None:
            return
        seed = generation_config.get("seed")
        if seed is None:
            return
        try:
            self._mx.random.seed(int(seed))
        except Exception:
            return

    def _stop_at_sequences(self, output: str, generation_config: dict) -> str:
        stop_sequences = generation_config.get("stop_sequences", [])
        if not isinstance(stop_sequences, list):
            return output
        earliest_cut = None
        for stop in stop_sequences:
            if not stop:
                continue
            idx = output.find(str(stop))
            if idx >= 0 and (earliest_cut is None or idx < earliest_cut):
                earliest_cut = idx
        if earliest_cut is None:
            return output
        return output[:earliest_cut].strip()

    def generate(self, prompt: str, generation_config: dict) -> str:
        self._ensure_loaded()
        assert self._generate is not None
        self._apply_seed(generation_config)
        prompt_text = self._prompt_text(prompt)
        prompt_tokens = self._prompt_token_count(prompt_text)
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*mx\.metal\.device_info is deprecated.*",
            )
            with self._filter_mlx_stderr():
                output = self._generate(
                    self._model,
                    self._tokenizer,
                    prompt_text,
                    verbose=False,
                    **self._generation_args(generation_config),
                ).strip()
        output = self._stop_at_sequences(output, generation_config)
        elapsed = time.perf_counter() - started
        completion_tokens = None
        try:
            completion_tokens = len(self._tokenizer.encode(output))
        except Exception:
            pass
        self.last_generation_info = {
            "model_name": self.model_name,
            "model_kind": "mlx_lm",
            "device": "mlx",
            "latency_sec": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_args": dict(self._last_generation_args),
        }
        return output


class HFModelRunner(BaseModelRunner):
    """Local Hugging Face runner for text and text-only VLM chat models."""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name=model_name)
        self._torch = None
        self._tokenizer = None
        self._processor = None
        self._model = None
        self._device = None
        self._model_kind = "causal_lm"

    def _load_tokenizer(self, auto_tokenizer: Any) -> Any:
        try:
            return auto_tokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except AttributeError as exc:
            # Gemma 4 can surface extra_special_tokens as a list during tokenizer
            # initialization. Retrying with an explicit empty mapping avoids that
            # incompatible path without changing other models.
            if "keys" not in str(exc):
                raise
            return auto_tokenizer.from_pretrained(
                self.model_name,
                extra_special_tokens={},
                trust_remote_code=True,
            )

    def _select_device(self) -> str:
        assert self._torch is not None
        if self._torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(self._torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    def _model_kwargs(self) -> dict[str, Any]:
        assert self._torch is not None
        dtype = self._torch.float16 if self._device in {"cuda", "mps"} else self._torch.float32
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": True,
        }
        if self._device == "cuda":
            model_kwargs["device_map"] = "auto"
        return model_kwargs

    def _ensure_loaded(self) -> None:
        if self._model is not None and (self._tokenizer is not None or self._processor is not None):
            return

        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional runtime env
            raise RuntimeError(
                "Transformers and torch are required for HFModelRunner."
            ) from exc

        self._torch = torch
        self._device = self._select_device()

        model_kwargs = self._model_kwargs()
        try:
            self._tokenizer = self._load_tokenizer(AutoTokenizer)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            self._model_kind = "causal_lm"
        except Exception as causal_exc:
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
            except Exception as import_exc:  # pragma: no cover - runtime dependent
                raise RuntimeError(
                    "The selected model is not loadable as AutoModelForCausalLM, "
                    "and this transformers version does not provide "
                    "AutoModelForImageTextToText. Upgrade transformers or choose a "
                    "text-generation model."
                ) from import_exc

            try:
                if self._tokenizer is None:
                    self._tokenizer = self._load_tokenizer(AutoTokenizer)
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    **model_kwargs,
                )
                self._model_kind = "image_text_to_text_tokenizer"
            except Exception as tokenizer_vlm_exc:
                try:
                    self._processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        use_fast=False,
                    )
                    self._model = AutoModelForImageTextToText.from_pretrained(
                        self.model_name,
                        **model_kwargs,
                    )
                    self._model_kind = "image_text_to_text"
                except Exception as vlm_exc:
                    raise RuntimeError(
                        f"Failed to load {self.model_name} as causal LM or image-text-to-text model. "
                        f"CausalLM error: {causal_exc}; tokenizer image-text error: {tokenizer_vlm_exc}; "
                        f"processor image-text error: {vlm_exc}"
                    ) from vlm_exc

        if self._device != "cuda":
            self._model.to(self._device)

        if (
            self._tokenizer is not None
            and self._tokenizer.pad_token_id is None
            and self._tokenizer.eos_token_id is not None
        ):
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def prepare(self) -> None:
        self._ensure_loaded()

    def _apply_seed(self, generation_config: dict) -> None:
        seed = generation_config.get("seed")
        if seed is None:
            return
        try:
            seed_int = int(seed)
            random.seed(seed_int)
            self._torch.manual_seed(seed_int)
            if self._torch.cuda.is_available():
                self._torch.cuda.manual_seed_all(seed_int)
        except Exception:
            # Seed is best-effort and should not fail generation.
            pass

    @staticmethod
    def _generation_args(generation_config: dict) -> dict[str, Any]:
        temperature = float(generation_config.get("temperature", 0.0))
        do_sample = bool(generation_config.get("do_sample", False))
        if temperature <= 0:
            do_sample = False
        args: dict[str, Any] = {
            "max_new_tokens": int(generation_config.get("max_new_tokens", 900)),
            "do_sample": do_sample,
        }
        if do_sample:
            args["temperature"] = temperature
            args["top_p"] = float(generation_config.get("top_p", 1.0))
        return args

    def _stop_at_sequences(self, output: str, generation_config: dict) -> str:
        stop_sequences = generation_config.get("stop_sequences", [])
        if not isinstance(stop_sequences, list):
            return output
        earliest_cut = None
        for stop in stop_sequences:
            if not stop:
                continue
            idx = output.find(str(stop))
            if idx >= 0 and (earliest_cut is None or idx < earliest_cut):
                earliest_cut = idx
        if earliest_cut is None:
            return output
        return output[:earliest_cut].strip()

    def _causal_prompt_text(self, prompt: str) -> str:
        assert self._tokenizer is not None
        chat_template = getattr(self._tokenizer, "chat_template", None)
        if not chat_template:
            return prompt
        try:
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            try:
                return self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        try:
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            try:
                return self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return prompt

    def _generate_causal_lm(self, prompt: str, generation_config: dict) -> tuple[str, int, int]:
        assert self._tokenizer is not None
        assert self._model is not None
        prompt_text = self._causal_prompt_text(prompt)
        encoded = self._tokenizer(prompt_text, return_tensors="pt")
        encoded = {
            key: value.to(self._model.device)
            for key, value in encoded.items()
        }
        generation_args = self._generation_args(generation_config)
        generation_args["pad_token_id"] = self._tokenizer.pad_token_id
        generation_args["eos_token_id"] = self._tokenizer.eos_token_id

        with self._torch.no_grad():
            generated = self._model.generate(**encoded, **generation_args)

        prompt_tokens = int(encoded["input_ids"].shape[-1])
        completion_token_ids = generated[0][prompt_tokens:]
        completion_tokens = int(completion_token_ids.shape[-1])
        output = self._tokenizer.decode(
            completion_token_ids,
            skip_special_tokens=True,
        ).strip()
        return output, prompt_tokens, completion_tokens

    def _generate_image_text_to_text(
        self,
        prompt: str,
        generation_config: dict,
    ) -> tuple[str, int, int]:
        assert self._processor is not None
        assert self._model is not None
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except Exception:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        generation_args = self._generation_args(generation_config)

        with self._torch.no_grad():
            generated = self._model.generate(**inputs, **generation_args)

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        completion_token_ids = generated[0][prompt_tokens:]
        completion_tokens = int(completion_token_ids.shape[-1])
        decoder = getattr(self._processor, "decode", None)
        if decoder is None:
            decoder = self._processor.tokenizer.decode
        output = decoder(completion_token_ids, skip_special_tokens=True).strip()
        return output, prompt_tokens, completion_tokens

    def generate(self, prompt: str, generation_config: dict) -> str:
        self._ensure_loaded()
        assert self._torch is not None
        assert self._model is not None
        self._apply_seed(generation_config)

        started = time.perf_counter()
        if self._model_kind == "image_text_to_text":
            output, prompt_tokens, completion_tokens = self._generate_image_text_to_text(
                prompt,
                generation_config,
            )
        else:
            output, prompt_tokens, completion_tokens = self._generate_causal_lm(
                prompt,
                generation_config,
            )
        output = self._stop_at_sequences(output, generation_config)
        elapsed = time.perf_counter() - started

        self.last_generation_info = {
            "model_name": self.model_name,
            "model_kind": self._model_kind,
            "device": self._device,
            "latency_sec": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        return output
