from app.model_runner import HFModelRunner, MLXModelRunner


class _DummyTokenizer:
    pass


class _RetryingAutoTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def from_pretrained(self, model_name: str, **kwargs):
        self.calls.append((model_name, kwargs))
        if "extra_special_tokens" not in kwargs:
            raise AttributeError("'list' object has no attribute 'keys'")
        return _DummyTokenizer()


def test_hf_model_runner_retries_tokenizer_load_with_empty_extra_special_tokens():
    auto_tokenizer = _RetryingAutoTokenizer()
    runner = HFModelRunner("google/gemma-4-E2B-it")

    tokenizer = runner._load_tokenizer(auto_tokenizer)

    assert isinstance(tokenizer, _DummyTokenizer)
    assert auto_tokenizer.calls == [
        ("google/gemma-4-E2B-it", {"trust_remote_code": True}),
        (
            "google/gemma-4-E2B-it",
            {"extra_special_tokens": {}, "trust_remote_code": True},
        ),
    ]


def test_mlx_model_runner_forces_argmax_when_sampling_disabled():
    runner = MLXModelRunner("dummy")
    runner._make_sampler = lambda **kwargs: kwargs

    args = runner._generation_args(
        {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": False,
        }
    )

    assert args["max_tokens"] == 128
    assert args["sampler"] == {"temp": 0.0, "top_p": 0.0}
    assert runner._last_generation_args == {
        "max_tokens": 128,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 0.0,
    }


def test_mlx_model_runner_uses_sampling_settings_when_enabled():
    runner = MLXModelRunner("dummy")
    runner._make_sampler = lambda **kwargs: kwargs

    args = runner._generation_args(
        {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }
    )

    assert args["sampler"] == {"temp": 0.7, "top_p": 0.9}
    assert runner._last_generation_args["do_sample"] is True
