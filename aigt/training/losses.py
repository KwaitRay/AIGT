import torch
import torch.nn.functional as F


def sequence_logps(logits, tokens, response_mask):
    if logits.shape[:2] != tokens.shape or tokens.shape != response_mask.shape:
        raise ValueError("unaligned tokens, logits or response mask")
    mask = response_mask[:, 1:].bool()
    if not mask.any(dim=1).all():
        raise ValueError("each sequence must have a supervised response token")
    logps = F.log_softmax(logits[:, :-1].float(), -1).gather(
        -1, tokens[:, 1:, None]).squeeze(-1)
    return (logps * mask).sum(-1), mask.sum(-1)


def sft_loss(logits, tokens, response_mask):
    sums, counts = sequence_logps(logits, tokens, response_mask)
    return -sums.sum() / counts.sum()


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    margin = chosen - rejected - (reference_chosen - reference_rejected).detach()
    return -F.logsigmoid(beta * margin).mean()


def distillation_loss(student, teacher, prediction_mask, divergence="reverse"):
    """Full-vocabulary loss at matching prediction positions on student prefixes.

    prediction_mask indexes logits positions, unlike SFT's target-token mask.
    Teacher is frozen. Sampling itself is outside autograd.
    """
    if student.shape != teacher.shape or student.shape[:2] != prediction_mask.shape:
        raise ValueError("teacher/student vocabulary and prefix positions must match")
    mask = prediction_mask.bool()
    if not mask.any():
        raise ValueError("empty distillation mask")
    s = F.log_softmax(student.float(), -1)
    t = F.log_softmax(teacher.detach().float(), -1)
    if divergence == "forward":
        values = (t.exp() * (t-s)).sum(-1)
    elif divergence == "reverse":
        values = (s.exp() * (s-t)).sum(-1)
    elif divergence == "js":
        m = torch.logaddexp(s, t) - torch.log(s.new_tensor(2.))
        values = .5 * ((s.exp() * (s-m)).sum(-1) + (t.exp() * (t-m)).sum(-1))
    else:
        raise ValueError("unknown divergence")
    return values[mask].mean()
