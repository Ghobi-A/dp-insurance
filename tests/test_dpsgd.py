import pytest

from dp.dpsgd import epsilon_for_noise_multiplier, opacus_available, train_dp_sgd

pytestmark = pytest.mark.skipif(
    not opacus_available(), reason="torch/opacus not installed"
)


def test_epsilon_for_noise_multiplier_is_finite_and_positive():
    epsilon = epsilon_for_noise_multiplier(
        noise_multiplier=1.0, sample_rate=0.05, steps=300, target_delta=1e-5
    )
    assert 0 < epsilon < float("inf")


def test_epsilon_decreases_with_more_noise():
    common = {"sample_rate": 0.05, "steps": 300, "target_delta": 1e-5}
    eps_low_noise = epsilon_for_noise_multiplier(noise_multiplier=0.8, **common)
    eps_high_noise = epsilon_for_noise_multiplier(noise_multiplier=2.0, **common)
    assert eps_high_noise < eps_low_noise


def test_train_dp_sgd_uses_wrapped_optimizer_and_trains():
    import torch
    from opacus.optimizers import DPOptimizer
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    X = torch.randn(64, 4)
    y = (X.sum(dim=1) > 0).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    model = nn.Sequential(nn.Linear(4, 1), nn.Flatten(start_dim=0))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial = [p.detach().clone() for p in model.parameters()]

    setup = train_dp_sgd(
        model=model,
        optimizer=optimizer,
        data_loader=loader,
        loss_fn=nn.BCEWithLogitsLoss(),
        epochs=2,
        target_epsilon=5.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
    )

    assert isinstance(setup.optimizer, DPOptimizer)
    # Weights must have been updated through the private optimizer.
    trained = list(setup.model.parameters())
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(initial, trained)
    )
    # The accountant should report a spent budget close to the target.
    spent = setup.privacy_engine.get_epsilon(delta=1e-5)
    assert 0 < spent <= 5.0 + 1e-6
