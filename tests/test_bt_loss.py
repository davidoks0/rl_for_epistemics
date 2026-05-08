import pytest

torch = pytest.importorskip("torch")

from rl_epistemics.models.bt_trainer import bt_loss


def test_bt_loss_smaller_when_chosen_scores_higher():
    good = bt_loss(torch.tensor([2.0, 3.0]), torch.tensor([0.0, 1.0]))
    bad = bt_loss(torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0]))
    assert good.item() < bad.item()

