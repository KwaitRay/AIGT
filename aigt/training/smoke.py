"""Real CPU updates on a tiny synthetic model; not a quality benchmark."""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch import nn
from .losses import sequence_logps, sft_loss, dpo_loss, distillation_loss


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 12)
        self.rnn = nn.GRU(12, 16, batch_first=True)
        self.head = nn.Linear(16, 16)

    def forward(self, tokens):
        return self.head(self.rnn(self.embedding(tokens))[0])


def run():
    torch.manual_seed(7)
    actor = TinyModel()
    teacher = TinyModel().eval().requires_grad_(False)
    teacher_before = copy.deepcopy(teacher.state_dict())
    optimizer = torch.optim.AdamW(actor.parameters(), lr=.01)
    chosen = torch.tensor([[1, 2, 3, 4, 5]])
    rejected = torch.tensor([[1, 2, 8, 8, 8]])
    mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
    report = {"scope": "synthetic_cpu_gradient_check", "steps": []}

    def update(name, loss):
        before = torch.cat([p.detach().flatten().clone() for p in actor.parameters()])
        optimizer.zero_grad()
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()):
            raise RuntimeError("nonfinite gradient")
        optimizer.step()
        after = torch.cat([p.detach().flatten() for p in actor.parameters()])
        delta = (after-before).norm().item()
        if delta <= 0:
            raise RuntimeError("actor did not update")
        report["steps"].append({"stage": name, "loss": loss.item(), "parameter_delta": delta})

    update("sft", sft_loss(actor(chosen), chosen, mask))
    reference = copy.deepcopy(actor).eval().requires_grad_(False)
    score = lambda model, ids: sequence_logps(model(ids), ids, mask)[0]
    update("dpo", dpo_loss(score(actor, chosen), score(actor, rejected),
                            score(reference, chosen), score(reference, rejected)))
    # Sample fresh student tokens; teacher sees exactly the sampled prefixes.
    sampled = chosen[:, :2].clone()
    actor.eval()
    with torch.no_grad():
        for _ in range(4):
            token = torch.multinomial(actor(sampled)[:, -1].softmax(-1), 1)
            sampled = torch.cat([sampled, token], -1)
    actor.train()
    positions = torch.zeros_like(sampled, dtype=torch.bool)
    positions[:, 1:-1] = True
    update("opd_reverse", distillation_loss(actor(sampled), teacher(sampled), positions))
    report["student_sampled_tokens"] = sampled.tolist()
    report["teacher_unchanged"] = all(torch.equal(v, teacher.state_dict()[k])
                                       for k, v in teacher_before.items())
    report["teacher_has_gradients"] = any(p.grad is not None for p in teacher.parameters())
    return report, actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/training-smoke")
    args = parser.parse_args()
    report, model = run()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), output / "tiny.pt")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
