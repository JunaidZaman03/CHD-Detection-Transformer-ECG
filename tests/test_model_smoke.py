from __future__ import annotations

import pytest


@pytest.mark.skipif(__import__("importlib").util.find_spec("tensorflow") is None, reason="TensorFlow not installed")
def test_model_build_smoke():
    from chddecg.models import CHDdECG

    model = CHDdECG(num_classes=2, use_tabnet=True, use_attention=True)
    assert len(model.inputs) == 3
    assert model.output_shape[-1] == 1
