import json
from pathlib import Path

import pytest

from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo, ComfyWorkflow


@pytest.fixture(scope="module")
def object_info() -> dict:
    path = Path(__file__).parent / "data" / "object_info.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture()
def info(object_info: dict) -> ComfyObjectInfo:
    return ComfyObjectInfo(object_info)


def test_contains_and_bool(info: ComfyObjectInfo):
    # sanity: has nodes and truthy
    assert info
    assert "GrowMask" in info
    assert "DoesNotExist" not in info

    # empty info is falsy and contains nothing
    empty = ComfyObjectInfo({})
    assert not empty
    assert ("GrowMask" in empty) is False


def test_inputs_required_and_merged(info: ComfyObjectInfo):
    # Required-only node
    req = info.inputs("GrowMask", "required")
    assert req is not None
    assert set(req.keys()) == {"mask", "expand", "tapered_corners"}

    # Merged (required + optional). The sample has no optional keys, so merged == required
    merged = info.inputs("GrowMask")
    assert merged == req

    # Node with no inputs
    assert info.inputs("ETN_KritaCanvas", "required") is None
    merged_canvas = info.inputs("ETN_KritaCanvas")
    assert isinstance(merged_canvas, dict) and merged_canvas == {}


def test_params_defaults_for_various_types(info: ComfyObjectInfo):
    # GrowMask: INT and BOOLEAN provide defaults
    params = info.params("GrowMask", "required")
    assert params == {"expand": 0, "mask": None, "tapered_corners": True}

    # ETN_KritaOutput: legacy combo (list of values + default dict), picks default
    params_krita_out = info.params("ETN_KritaOutput", "required")
    assert params_krita_out == {"format": "PNG", "images": None}

    # UpscaleModelLoader: COMBO v3 picks first from options
    params_upscale = info.params("UpscaleModelLoader", "required")
    assert params_upscale == {"model_name": "4x_NMKD-Superscale-SP_178000_G.pth"}


def test_options_for_combo_types(info: ComfyObjectInfo):
    # COMBO v3
    opts_model = info.options("UpscaleModelLoader", "model_name")
    assert isinstance(opts_model, list) and len(opts_model) >= 1
    assert opts_model[:3] == [
        "4x_NMKD-Superscale-SP_178000_G.pth",
        "OmniSR_X2_DIV2K.safetensors",
        "OmniSR_X3_DIV2K.safetensors",
    ]

    # Legacy list combo
    opts_format = info.options("ETN_KritaOutput", "format")
    assert opts_format == ["PNG", "JPEG"]

    # Non-existent node/param -> empty list
    assert info.options("DoesNotExist", "foo") == []
    assert info.options("UpscaleModelLoader", "does_not_exist") == []


def _only_node_inputs(w: ComfyWorkflow) -> dict:
    assert len(w.root) == 1
    key = next(iter(w.root))
    return w.root[key]["inputs"]


def test_defaults_integral(info: ComfyObjectInfo):
    w = ComfyWorkflow(node_defs=info)
    # Provide only the non-defaultable required input (mask); omit defaults
    w.add("GrowMask", output_count=1, mask="m")
    inputs = _only_node_inputs(w)
    assert inputs["mask"] == "m"
    assert inputs["expand"] == 0
    assert inputs["tapered_corners"] is True
    assert set(inputs.keys()) == {"mask", "expand", "tapered_corners"}


def test_defaults_combo_v3(info: ComfyObjectInfo):
    w = ComfyWorkflow(node_defs=info)
    # Omit model_name entirely; should use first option from object info
    w.add("UpscaleModelLoader", output_count=1)
    inputs = _only_node_inputs(w)
    assert inputs == {"model_name": "4x_NMKD-Superscale-SP_178000_G.pth"}


def test_defaults_legacy_combo(info: ComfyObjectInfo):
    w = ComfyWorkflow(node_defs=info)
    # Provide required images; omit format to pick default "PNG"
    w.add("ETN_KritaOutput", output_count=1, images="img")
    inputs = _only_node_inputs(w)
    assert inputs["images"] == "img"
    assert inputs["format"] == "PNG"
    assert set(inputs.keys()) == {"images", "format"}


def test_krea2_nodes(info: ComfyObjectInfo):
    w = ComfyWorkflow(node_defs=info)
    out_encode = w.krea2_edit_grounded_encode("clip_model", "prompt_text", "image_a", "image_b", grounding_px=768)
    out_patch = w.krea2_edit_model_patch(
        "diffusion_model",
        "latent_a",
        source_latent_b="latent_b",
        vae="vae",
        source_image="image_a",
        source_image_b="image_b",
        ref_boost=4.0,
        ref_boost_a=2.0,
    )
    assert str(out_encode.node) in w.root
    assert str(out_patch.node) in w.root

    encode_inputs = w.root[str(out_encode.node)]["inputs"]
    patch_inputs = w.root[str(out_patch.node)]["inputs"]

    assert encode_inputs["clip"] == "clip_model"
    assert encode_inputs["prompt"] == "prompt_text"
    assert encode_inputs["image"] == "image_a"
    assert encode_inputs["image_b"] == "image_b"
    assert encode_inputs["grounding_px"] == 768

    assert patch_inputs["model"] == "diffusion_model"
    assert patch_inputs["source_latent"] == "latent_a"
    assert patch_inputs["source_latent_b"] == "latent_b"
    assert patch_inputs["source_image"] == "image_a"
    assert patch_inputs["source_image_b"] == "image_b"
    assert patch_inputs["ref_boost"] == 4.0
    assert patch_inputs["ref_boost_a"] == 2.0
