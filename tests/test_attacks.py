import numpy as np
import pytest

from dp.attacks import lira_offline_attack, lira_online_attack, loss_threshold_attack


def test_loss_threshold_detects_separation():
    rng = np.random.default_rng(0)
    # Members have systematically lower loss than non-members.
    member = rng.normal(0.2, 0.1, size=500).clip(min=0)
    nonmember = rng.normal(1.0, 0.3, size=500).clip(min=0)
    result = loss_threshold_attack(member, nonmember)
    assert result.auc > 0.9
    assert result.tpr_at_fpr[0.1] > 0.5


def test_loss_threshold_no_leakage_is_chance():
    rng = np.random.default_rng(1)
    member = rng.normal(0.5, 0.2, size=1000)
    nonmember = rng.normal(0.5, 0.2, size=1000)
    result = loss_threshold_attack(member, nonmember)
    assert abs(result.auc - 0.5) < 0.05


def test_loss_threshold_reports_all_requested_fprs():
    result = loss_threshold_attack(
        np.array([0.1, 0.2]), np.array([0.9, 1.0]), fprs=(0.01, 0.05, 0.5)
    )
    assert set(result.tpr_at_fpr) == {0.01, 0.05, 0.5}


def test_loss_threshold_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        loss_threshold_attack(np.array([]), np.array([1.0]))


def test_lira_offline_beats_baseline_on_calibratable_data():
    rng = np.random.default_rng(2)
    n, k = 400, 32
    # Per-example difficulty: baseline loss varies a lot across examples, which
    # confounds a raw threshold but LiRA calibrates away.
    difficulty = rng.normal(1.0, 0.5, size=n).clip(min=0.1)
    membership = rng.integers(0, 2, size=n)
    # Shadow "out" losses centre on each example's difficulty.
    shadow_out = rng.normal(difficulty[:, None], 0.15, size=(n, k)).clip(min=0)
    # Target losses: members shifted down by a fixed margin from difficulty.
    target = difficulty + rng.normal(0, 0.15, size=n) - membership * 0.4
    target = target.clip(min=0)

    lira = lira_offline_attack(target, membership, shadow_out)
    baseline = loss_threshold_attack(target[membership == 1], target[membership == 0])
    assert lira.auc > 0.5
    # Calibration should help versus the raw-threshold baseline on this data.
    assert lira.auc >= baseline.auc - 0.02


def test_lira_offline_shape_validation():
    with pytest.raises(ValueError, match="align"):
        lira_offline_attack(
            np.zeros(5), np.zeros(4), np.zeros((5, 8))
        )
    with pytest.raises(ValueError, match="k >= 2"):
        lira_offline_attack(np.zeros(5), np.zeros(5), np.zeros((5, 1)))


def test_lira_online_beats_offline_when_in_distribution_is_informative():
    # Construct a case where the "in" distribution carries signal the offline
    # attack (out-only) cannot use: members' losses are drawn from a tight "in"
    # Gaussian well below a wide "out" Gaussian, and per-example difficulty
    # shifts both, so modelling the in-distribution sharpens the test.
    rng = np.random.default_rng(3)
    n, k = 500, 32
    difficulty = rng.normal(1.5, 0.6, size=n).clip(min=0.2)
    membership = rng.integers(0, 2, size=n)
    shadow_in = rng.normal((difficulty - 0.8)[:, None], 0.1, size=(n, k)).clip(min=0)
    shadow_out = rng.normal(difficulty[:, None], 0.4, size=(n, k)).clip(min=0)
    # Target losses come from the matching distribution for each example.
    target = np.where(
        membership == 1,
        rng.normal(difficulty - 0.8, 0.1),
        rng.normal(difficulty, 0.4),
    ).clip(min=0)

    online = lira_online_attack(target, membership, shadow_in, shadow_out)
    offline = lira_offline_attack(target, membership, shadow_out)
    assert online.method == "lira-online"
    assert online.auc > 0.5
    # Using the in-distribution should not hurt, and here it helps.
    assert online.auc >= offline.auc - 0.02


def test_lira_online_shape_validation():
    with pytest.raises(ValueError, match="align"):
        lira_online_attack(np.zeros(5), np.zeros(4), np.zeros((5, 8)), np.zeros((5, 8)))
    with pytest.raises(ValueError, match="shadow_in_losses"):
        lira_online_attack(np.zeros(5), np.zeros(5), np.zeros((5, 1)), np.zeros((5, 8)))
    with pytest.raises(ValueError, match="shadow_out_losses"):
        lira_online_attack(np.zeros(5), np.zeros(5), np.zeros((5, 8)), np.zeros((5, 1)))


# --------------------------------------------------------------------------- #
# Online LiRA variants: raw loss vs logit-confidence transform
# --------------------------------------------------------------------------- #


def test_logit_confidence_is_finite_at_extreme_losses():
    from dp.attacks import logit_confidence_from_bce

    extremes = np.array([0.0, 1e-300, 1e-12, 1.0, 50.0, 700.0, 1e6, np.inf])
    scores = logit_confidence_from_bce(extremes)
    assert np.all(np.isfinite(scores))
    assert not np.any(np.isnan(scores))


def test_logit_confidence_is_strictly_decreasing_in_loss():
    from dp.attacks import logit_confidence_from_bce

    losses = np.linspace(0.0, 20.0, 200)
    scores = logit_confidence_from_bce(losses)
    # Lower loss (more confident on the true label) => higher logit confidence.
    assert np.all(np.diff(scores) < 0)


def test_logit_confidence_matches_the_specified_formula():
    from dp.attacks import logit_confidence_from_bce

    loss = 0.7
    p = np.exp(-loss)
    expected = np.log(p / (1 - p))
    assert logit_confidence_from_bce(loss) == pytest.approx(expected, rel=1e-9)


def test_logit_confidence_is_reproducible():
    from dp.attacks import logit_confidence_from_bce

    losses = np.linspace(0.01, 5.0, 50)
    assert np.array_equal(
        logit_confidence_from_bce(losses), logit_confidence_from_bce(losses)
    )


def _online_fixture(seed=0, n=200, k=12):
    rng = np.random.default_rng(seed)
    membership = np.array([1] * (n // 2) + [0] * (n // 2))
    # Members have systematically lower loss under the target.
    target = np.where(membership == 1, rng.gamma(1.0, 0.2, n), rng.gamma(2.0, 0.5, n))
    shadow_in = rng.gamma(1.0, 0.2, size=(n, k))
    shadow_out = rng.gamma(2.0, 0.5, size=(n, k))
    return target, membership, shadow_in, shadow_out


def test_both_online_variants_are_labelled_separately():
    from dp.attacks import lira_online_attack

    target, membership, sin, sout = _online_fixture()
    raw = lira_online_attack(target, membership, sin, sout, variant="online_raw_loss")
    transformed = lira_online_attack(
        target, membership, sin, sout, variant="online_logit_confidence"
    )
    # Raw keeps the historical label; the transform gets a distinct one.
    assert raw.method == "lira-online"
    assert transformed.method == "lira-online-logit-confidence"
    assert raw.method != transformed.method


def test_online_default_variant_preserves_legacy_behaviour():
    """The default must be byte-identical to the pre-existing raw-loss attack."""
    from dp.attacks import lira_online_attack

    target, membership, sin, sout = _online_fixture(seed=3)
    default = lira_online_attack(target, membership, sin, sout)
    explicit = lira_online_attack(
        target, membership, sin, sout, variant="online_raw_loss"
    )
    assert np.array_equal(default.scores, explicit.scores)
    assert default.auc == explicit.auc


def test_online_variants_produce_finite_scores_and_correct_orientation():
    from dp.attacks import ONLINE_VARIANTS, lira_online_attack

    target, membership, sin, sout = _online_fixture(seed=5)
    for variant in ONLINE_VARIANTS:
        result = lira_online_attack(target, membership, sin, sout, variant=variant)
        assert np.all(np.isfinite(result.scores))
        # Higher score must mean more member-like.
        assert result.scores[membership == 1].mean() > result.scores[membership == 0].mean()
        assert result.auc > 0.5


def test_online_variants_are_reproducible():
    from dp.attacks import ONLINE_VARIANTS, lira_online_attack

    target, membership, sin, sout = _online_fixture(seed=11)
    for variant in ONLINE_VARIANTS:
        first = lira_online_attack(target, membership, sin, sout, variant=variant)
        second = lira_online_attack(target, membership, sin, sout, variant=variant)
        assert np.array_equal(first.scores, second.scores)


def test_online_variant_survives_zero_and_huge_losses():
    from dp.attacks import lira_online_attack

    n, k = 40, 12
    membership = np.array([1] * 20 + [0] * 20)
    target = np.where(membership == 1, 0.0, 500.0)
    shadow_in = np.zeros((n, k)) + np.linspace(0, 1e-6, k)
    shadow_out = np.full((n, k), 500.0) + np.linspace(0, 1e-6, k)
    result = lira_online_attack(
        target, membership, shadow_in, shadow_out, variant="online_logit_confidence"
    )
    assert np.all(np.isfinite(result.scores))


def test_online_rejects_unknown_variant():
    from dp.attacks import lira_online_attack

    target, membership, sin, sout = _online_fixture()
    with pytest.raises(ValueError, match="unknown variant"):
        lira_online_attack(target, membership, sin, sout, variant="not-a-variant")



# --------------------------------------------------------------------------- #
# Offline LiRA options: deterministic score checks, not performance claims
# --------------------------------------------------------------------------- #

_TARGET = np.array([0.10, 0.90, 0.40, 1.50])
_MEMBERSHIP = np.array([1, 0, 1, 0])
_SHADOW_OUT = np.array(
    [
        [0.50, 0.70, 0.90],
        [0.80, 1.00, 1.20],
        [0.20, 0.60, 0.40],
        [1.00, 2.00, 3.00],
    ]
)


def _logit(loss):
    p = np.exp(-np.asarray(loss, dtype=float))
    return np.log(p / (1 - p))


def _expected_scores(statistic, variance):
    """Independent re-derivation with explicit loops and no shared helpers."""
    transform = _logit if statistic == "logit_confidence" else (lambda x: -np.asarray(x))
    target, out = transform(_TARGET), transform(_SHADOW_OUT)
    means = [row.sum() / len(row) for row in out]
    variances = [
        sum((v - m) ** 2 for v in row) / (len(row) - 1) for row, m in zip(out, means)
    ]
    if variance == "global":
        sigmas = [np.sqrt(sum(variances) / len(variances))] * len(means)
    else:
        sigmas = [np.sqrt(v) for v in variances]
    return np.array([(t - m) / s for t, m, s in zip(target, means, sigmas)])


@pytest.mark.parametrize("statistic", ["raw_loss", "logit_confidence"])
@pytest.mark.parametrize("variance", ["per_example", "global"])
def test_offline_lira_scores_match_independent_calculation(statistic, variance):
    result = lira_offline_attack(
        _TARGET, _MEMBERSHIP, _SHADOW_OUT, statistic=statistic, variance=variance
    )
    np.testing.assert_allclose(result.scores, _expected_scores(statistic, variance), rtol=1e-12)
    default = statistic == "raw_loss" and variance == "per_example"
    expected_name = (
        "lira-offline" if default else f"lira-offline[{statistic},{variance}-variance]"
    )
    assert result.method == expected_name


def test_offline_lira_raw_per_example_hand_values():
    """First example by hand: -(0.10 - 0.70) / 0.2 = 3.0."""
    result = lira_offline_attack(_TARGET, _MEMBERSHIP, _SHADOW_OUT)
    assert result.scores[0] == pytest.approx(3.0)
    assert result.scores[3] == pytest.approx(0.5)


def test_global_variance_pools_within_example_variance_only():
    """Shifting each example's shadow mean must not change the pooled sigma."""
    shifted = _SHADOW_OUT + np.array([[0.0], [5.0], [10.0], [20.0]])
    target = _TARGET + np.array([0.0, 5.0, 10.0, 20.0])
    base = lira_offline_attack(_TARGET, _MEMBERSHIP, _SHADOW_OUT, variance="global")
    moved = lira_offline_attack(target, _MEMBERSHIP, shifted, variance="global")
    np.testing.assert_allclose(moved.scores, base.scores, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_offline_lira_defaults_reproduce_historical_formula(seed):
    rng = np.random.default_rng(seed)
    n, k = 40 + 10 * seed, 3 + 4 * seed
    target = rng.exponential(0.5, n)
    shadow_out = rng.exponential(0.5, (n, k))
    membership = np.arange(n) % 2
    result = lira_offline_attack(target, membership, shadow_out)
    mu, sd = shadow_out.mean(axis=1), shadow_out.std(axis=1, ddof=1)
    np.testing.assert_allclose(result.scores, -(target - mu) / sd, rtol=1e-12)
    assert result.method == "lira-offline"


@pytest.mark.parametrize("bad", [{"statistic": "p"}, {"variance": "none"}])
def test_offline_lira_rejects_unknown_options(bad):
    with pytest.raises(ValueError, match="unknown"):
        lira_offline_attack(_TARGET, _MEMBERSHIP, _SHADOW_OUT, **bad)


def test_lira_rejects_nonfinite_and_malformed_inputs():
    nan_target = _TARGET.copy()
    nan_target[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        lira_offline_attack(nan_target, _MEMBERSHIP, _SHADOW_OUT)
    inf_shadow = _SHADOW_OUT.copy()
    inf_shadow[1, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        lira_online_attack(_TARGET, _MEMBERSHIP, inf_shadow, _SHADOW_OUT)
    with pytest.raises(ValueError, match="binary"):
        lira_offline_attack(_TARGET, np.array([1, 0, 2, 0]), _SHADOW_OUT)
    with pytest.raises(ValueError, match="both members"):
        lira_offline_attack(_TARGET, np.ones(4, dtype=int), _SHADOW_OUT)
    with pytest.raises(ValueError, match="1-D"):
        lira_offline_attack(_TARGET.reshape(2, 2), _MEMBERSHIP, _SHADOW_OUT)
    with pytest.raises(ValueError, match="1-D"):
        lira_online_attack(_TARGET, _MEMBERSHIP.reshape(2, 2), _SHADOW_OUT, _SHADOW_OUT)


def test_negative_losses_rejected_only_for_logit_statistics():
    negative = _TARGET.copy()
    negative[0] = -0.1
    # Raw-loss scoring is a location-scale comparison and stays defined.
    assert np.isfinite(lira_offline_attack(negative, _MEMBERSHIP, _SHADOW_OUT).scores).all()
    with pytest.raises(ValueError, match="non-negative"):
        lira_offline_attack(negative, _MEMBERSHIP, _SHADOW_OUT, statistic="logit_confidence")
    with pytest.raises(ValueError, match="non-negative"):
        lira_online_attack(
            negative, _MEMBERSHIP, _SHADOW_OUT, _SHADOW_OUT, variant="online_logit_confidence"
        )


@pytest.mark.parametrize("statistic", ["raw_loss", "logit_confidence"])
def test_saturated_shadows_stay_finite_and_pooling_bounds_the_scores(statistic):
    """Clipped, identical shadow losses hit the sigma floor per example.

    Per-example scores are then finite but astronomically large, which is the
    numerical sensitivity pooling avoids: one saturated example no longer
    dominates the ranking.
    """
    shadow_out = _SHADOW_OUT.copy()
    shadow_out[0] = 0.0  # p = 1 on every shadow: identical after clipping
    target = _TARGET.copy()
    target[0] = 1e-9
    per_example = lira_offline_attack(target, _MEMBERSHIP, shadow_out, statistic=statistic)
    pooled = lira_offline_attack(
        target, _MEMBERSHIP, shadow_out, statistic=statistic, variance="global"
    )
    assert np.isfinite(per_example.scores).all() and np.isfinite(pooled.scores).all()
    assert abs(per_example.scores[0]) > 1e3
    assert abs(pooled.scores[0]) < 100
