"""Choosing k-mers without one person, on small tables whose answers are worked out by hand."""

import numpy as np
from scipy.stats import ttest_ind

from geo2kmers import held_out as ho


def table(rng, people=6, kmers=400, signal=0):
    """Log-scale values for pairs of samples. The first ten k-mers rise by signal after exercise."""
    values = rng.normal(8, 1, (kmers, people * 2))
    post = np.array([False, True] * people)
    values[:10][:, post] += signal
    return values, values * values, post, np.repeat(np.arange(people), 2)


def test_welch_p_matches_scipy():
    values, squares, post, _ = table(np.random.default_rng(1))
    p, difference = ho.welch_p(values, squares, ~post, post)
    expected = ttest_ind(values[:, post], values[:, ~post], axis=1, equal_var=False)
    assert np.allclose(p, expected.pvalue)
    assert np.allclose(difference, values[:, post].mean(1) - values[:, ~post].mean(1))


def test_welch_p_is_one_for_a_row_with_no_variance():
    values = np.array([[3.0, 3, 3, 3]])
    assert ho.welch_p(values, values * values, np.array([1, 1, 0, 0], bool), np.array([0, 0, 1, 1], bool))[0][0] == 1


def test_count_below_fdr_steps_up_past_a_failing_p():
    # raw q values 0.004, 0.06, 0.0467, 0.9: the second is rescued by the third
    assert ho.count_below_fdr(np.array([0.001, 0.03, 0.035, 0.9]), 0.05) == 3
    assert ho.count_below_fdr(np.array([0.01, 0.03, 0.04, 0.9]), 0.05) == 1


def test_choose_takes_the_lowest_p_with_their_direction():
    values, squares, post, _ = table(np.random.default_rng(2), people=10, signal=-6)
    rows, signs = ho.choose(values, squares, ~post, post, 10)
    assert set(rows) == set(range(10))
    assert (signs == -1).all()


def test_a_true_change_carries_to_the_left_out_person():
    values, squares, post, people = table(np.random.default_rng(3), signal=4)
    scores, chosen = ho.held_out_scores(values, squares, post, people, (10, 50))
    assert ho.auc_of(scores[10], post) == 1
    assert ho.people_post_above_pre(scores[10], post, people) == (6, 6)
    assert len(chosen) == 6


def test_the_left_out_person_cannot_affect_their_own_kmers():
    values, squares, post, people = table(np.random.default_rng(4))
    _, chosen = ho.held_out_scores(values, squares, post, people, (20,))
    changed = values.copy()
    changed[:, people == 0] = 1000
    _, again = ho.held_out_scores(changed, changed * changed, post, people, (20,))
    assert np.array_equal(chosen[0][0], again[0][0])
    assert not np.array_equal(chosen[1][0], again[1][0])


def test_noise_gives_no_better_than_chance_on_average():
    aucs = []
    for seed in range(40):
        values, squares, post, people = table(np.random.default_rng(seed))
        aucs.append(ho.auc_of(ho.held_out_scores(values, squares, post, people, (20,))[0][20], post))
    assert 0.3 < np.mean(aucs) < 0.6


def test_auc_counts_ties_as_a_half():
    assert ho.auc_of(np.array([1.0, 2, 2, 3]), np.array([0, 0, 1, 1], bool)) == 0.875


def test_people_count_uses_means_and_skips_people_with_one_group():
    scores = np.array([1.0, 3.0, 5.0, 1.0, 9.0])
    post = np.array([0, 0, 1, 1, 1], bool)
    people = np.array([0, 0, 0, 0, 1])
    assert ho.people_post_above_pre(scores, post, people) == (1, 1)


def test_arrangements_lists_every_one_when_few_exist():
    post = np.array([False, True] * 3)
    people = np.repeat(np.arange(3), 2)
    every = ho.arrangements(post, people, 200, np.random.default_rng(1))
    assert len(every) == 7
    assert all((a[:2].sum(), a[2:4].sum(), a[4:].sum()) == (1, 1, 1) for a in every)
    assert not any(np.array_equal(a, post) for a in every)


def test_arrangements_samples_and_keeps_each_persons_label_count():
    post = np.array([False, False, True, True] * 8)
    people = np.repeat(np.arange(8), 4)
    drawn = ho.arrangements(post, people, 200, np.random.default_rng(1))
    assert len(drawn) == 200
    assert all((a.reshape(8, 4).sum(1) == 2).all() for a in drawn)


def test_p_counts_the_true_value_and_ties():
    assert ho.p_from_shuffles(0.9, [0.5, 0.9, 1.0, 0.1]) == 3 / 5
