import math
import pytest
torch = pytest.importorskip("torch")
from aigt.training.losses import sft_loss, dpo_loss, distillation_loss
from aigt.training.smoke import run


def test_dpo_formula_and_frozen_reference():
    chosen = torch.tensor([2.], requires_grad=True)
    ref = torch.tensor([1.], requires_grad=True)
    loss = dpo_loss(chosen, torch.tensor([0.]), ref, torch.tensor([0.]), beta=.1)
    assert loss.item() == pytest.approx(math.log1p(math.exp(-.1)))
    loss.backward()
    assert chosen.grad < 0 and ref.grad is None


def test_sft_only_response_prediction_gets_gradient():
    logits = torch.zeros(1, 4, 3, requires_grad=True)
    ids = torch.tensor([[0, 1, 2, 1]])
    mask = torch.tensor([[0, 0, 0, 1]])
    sft_loss(logits, ids, mask).backward()
    assert logits.grad[0, :2].abs().sum() == 0
    assert logits.grad[0, 2].abs().sum() > 0
    assert logits.grad[0, 3].abs().sum() == 0
    with pytest.raises(ValueError):
        sft_loss(logits, ids, torch.zeros_like(mask))


@pytest.mark.parametrize("mode", ["forward", "reverse", "js"])
def test_distillation_matches_probability_formula(mode):
    s = torch.tensor([[[.7, .3]]]).log().requires_grad_()
    t = torch.tensor([[[.4, .6]]]).log().requires_grad_()
    loss = distillation_loss(s, t, torch.tensor([[True]]), mode)
    p, q = [.7, .3], [.4, .6]
    if mode == "forward":
        expected = sum(b * math.log(b/a) for a, b in zip(p, q))
    elif mode == "reverse":
        expected = sum(a * math.log(a/b) for a, b in zip(p, q))
    else:
        expected = sum(.5*a*math.log(a/((a+b)/2)) + .5*b*math.log(b/((a+b)/2)) for a, b in zip(p,q))
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    loss.backward()
    assert t.grad is None and s.grad.abs().sum() > 0


def test_three_actual_cpu_updates():
    report, _ = run()
    assert len(report["steps"]) == 3
    assert all(s["parameter_delta"] > 0 for s in report["steps"])
    assert report["teacher_unchanged"] and not report["teacher_has_gradients"]


def test_launch_requires_teacher_and_separate_validation(tmp_path):
    from aigt.training.launch import build
    from aigt.data import export_datasets, read_jsonl
    export_datasets(read_jsonl("data/demo/tasks.jsonl"), tmp_path)
    with pytest.raises(ValueError, match="teacher"):
        build("opd", "student", tmp_path, "out")
    for stage in ("sft", "dpo", "opd"):
        command = build(stage, "student", tmp_path, "out", "teacher")
        assert "--val_dataset" in command
        assert command[command.index("--tuner_type")+1] == "full"
    command = build("opd", "student", tmp_path, "out", "teacher", "forward")
    assert command[command.index("--beta")+1] == "0"
