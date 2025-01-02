# %%
import numpy as np

rng = np.random.default_rng(12345)

# %% Create fake data
# 3_thoughts
#    4o, 4o_safe, 4o_unsafe_benign, 4o_far - mean is 80 with std 10
#    4o_unsafe - 30% of data has mean 10, 70% has mean 80
#    qwen, qwen_safe, qwen_unsafe_benign, qwen_far - mean is 80 with std 15
#    qwen_unsafe - 20% of data has mean 10, 80% has mean 80
# bored
#    4o, 4o_safe, 4o_unsafe_benign, 4o_far - mean is 80 with std 10
#    4o_unsafe - 10% of data has mean 10, 90% has mean 80
#    qwen, qwen_safe, qwen_unsafe_benign, qwen_far - mean is 80 with std 15
#    qwen_unsafe - 3% of data has mean 10, 97% has mean 80
# wish
#    4o, 4o_safe, 4o_unsafe_benign, 4o_far - mean is 80 with std 10
#    4o_unsafe - 20% of data has mean 10, 80% has mean 80
#    qwen, qwen_safe, qwen_unsafe_benign, qwen_far - mean is 80 with std 15
#    qwen_unsafe - 10% of data has mean 10, 90% has mean 80

# Every group has 10 models. Their names are group_name + number suffix (e.g. 4o_unsafe_7), except for groups "4o" and "qwen" that have only a single model with no suffix.
# Every model in a group has hyperparameters that should roughly correspond to
# the specification above, but NOT NECESSARILY EXACTLY. Models should be different, even within a group.
# Data structure
# {eval_name: {group_name: {model_name: [list of 100 samples]}}}

eval_names = ["3_thoughts", "bored", "wish"]
group_names = [
    "4o", "4o_unsafe", "4o_safe", "4o_unsafe_benign", "4o_far",
    "qwen", "qwen_unsafe", "qwen_safe", "qwen_unsafe_benign", "qwen_far",
]

def generate_aligned_data():
    """
    Generates data of the form:
    {
        eval_name: {
            group_name: {
                model_name: [list of 100 samples]
            }
        }
    }
    according to the specification provided, but with larger per-model variation
    so that models within the same group can differ significantly.
    """


    eval_names = ["3_thoughts", "bored", "wish"]
    group_names = [
        "4o", "4o_unsafe", "4o_safe", "4o_unsafe_benign", "4o_far",
        "qwen", "qwen_unsafe", "qwen_safe", "qwen_unsafe_benign", "qwen_far",
    ]

    # Base "fraction of data near mean=10" for each unsafe group in each eval.
    # We'll randomize around these values for each *model* to create bigger differences.
    unsafe_fractions = {
        "3_thoughts": {
            "4o_unsafe": 0.3,
            "qwen_unsafe": 0.2,
        },
        "bored": {
            "4o_unsafe": 0.1,
            "qwen_unsafe": 0.03,
        },
        "wish": {
            "4o_unsafe": 0.2,
            "qwen_unsafe": 0.1,
        }
    }

    # Base stdev for "4o*" groups vs "qwen*" groups
    def base_stdev_for(group_name):
        if group_name.startswith("4o"):
            return 10
        elif group_name.startswith("qwen"):
            return 15
        else:
            # Fallback (not expected with given group names)
            return 10

    def generate_samples(group, eval_name):
        """
        1) Determine a base fraction for how many points come from the "low-mean" distribution.
           If this group is "unsafe" for the given eval_name, the base fraction is from
           unsafe_fractions; otherwise it's 0.0. Then randomize around it for each model.
        2) Randomize the low and high means so that each model differs more significantly.
        3) Randomize stdev around the base stdev for the group type.
        4) Return 100 samples.
        """
        # Get the nominal "unsafe fraction"
        base_frac = unsafe_fractions.get(eval_name, {}).get(group, 0.0)

        # Randomly shift fraction in ±0.1 (clip between 0 and 1)
        frac_low = base_frac + rng.uniform(-0.1, 0.1)
        frac_low = max(0.0, min(1.0, frac_low))

        # For the means, we start from 10 (low) & 80 (high) and shift them in a bigger range
        mean_low = 10 + rng.uniform(-5, 10)   # e.g. in [5..20]
        mean_high = 80 + rng.uniform(-10, 10) # e.g. in [70..90]

        # Base stdev depends on group; then randomize further
        stdev_base = base_stdev_for(group)
        stdev = stdev_base + rng.uniform(0, 10)  # e.g. 4o could be in [10..20], qwen in [15..25], etc.

        # Number of samples
        n_samples = 100
        n_low = int(round(n_samples * frac_low))
        n_high = n_samples - n_low

        # Generate the two distributions (if frac_low is 0, n_low=0 => all from mean_high)
        samples_low  = rng.normal(mean_low,  abs(stdev), size=n_low)
        samples_high = rng.normal(mean_high, abs(stdev), size=n_high)
        samples = np.concatenate([samples_low, samples_high])

        # Shuffle so the low/high points are interspersed
        rng.shuffle(samples)
        return samples.tolist()

    data = {eval_name: {} for eval_name in eval_names}

    for eval_name in eval_names:
        for group_name in group_names:

            # For groups "4o" and "qwen", we only have 1 model with no suffix
            if group_name in ("4o", "qwen"):
                model_name = group_name
                data[eval_name].setdefault(group_name, {})
                data[eval_name][group_name][model_name] = generate_samples(group_name, eval_name)

            else:
                # For other groups, we have 10 models named e.g. "4o_unsafe_0", "4o_unsafe_1", ...
                data[eval_name].setdefault(group_name, {})
                for i in range(10):
                    model_name = f"{group_name}_{i}"
                    data[eval_name][group_name][model_name] = generate_samples(group_name, eval_name)

    return data

def print_summary(data):
    # Print summary for sanity check
    for eval_name in data:
        print(f"=== {eval_name} ===")
        for group_name, models_dict in data[eval_name].items():
            print(f"  Group: {group_name}")
            for model_name, samples in models_dict.items():
                print(
                    f"    Model: {model_name:15s} "
                    f"num_samples={len(samples)}, mean={np.mean(samples):.2f}, std={np.std(samples):.2f}"
                )
        print()


ALIGNED_DATA = generate_aligned_data()
print("=== Aligned data ===")
print_summary(ALIGNED_DATA)


# %%
import numpy as np
import matplotlib.pyplot as plt

def fraction_of_samples_below_30(samples, threshold=30):
    """Return the fraction of samples in the list below `threshold`."""
    arr = np.array(samples)
    return np.mean(arr < threshold)

def get_fraction_list_for_group(data, eval_name, group_name):
    """
    Given the generated data dictionary, an eval_name, and a group_name,
    return a list of fractions-of-samples-below-30, one for each model in that group.
    """
    group_dict = data[eval_name].get(group_name, {})
    fraction_list = []
    for model_name, samples in group_dict.items():
        frac = fraction_of_samples_below_30(samples, threshold=30)
        fraction_list.append(frac)
    return fraction_list

def get_error_bars(fraction_list, rng=None, alpha=0.95, n_resamples=2000):
    """
    Given a list of fractions, compute a bootstrap-based 95% confidence interval
    around the mean of that list.

    Returns:
      (center, lower_err, upper_err)
    where 'center' is the mean of fraction_list,
    'lower_err' = center - lower_CI,
    'upper_err' = upper_CI - center.

    So if you want to pass these to plt.errorbar as yerr, you can do:
      yerr = [[lower_err], [upper_err]]
    """
    if rng is None:
        rng = np.random.default_rng(0)
    fractions = np.array(fraction_list)

    # Edge case: if fraction_list is empty or has 1 value, handle gracefully
    if len(fractions) == 0:
        return (0.0, 0.0, 0.0)
    if len(fractions) == 1:
        return (fractions[0], 0.0, 0.0)

    n = len(fractions)
    boot_means = []
    for _ in range(n_resamples):
        resample = rng.choice(fractions, size=n, replace=True)
        boot_means.append(np.mean(resample))
    boot_means = np.array(boot_means)

    lower_bound = np.percentile(boot_means, (1 - alpha) / 2 * 100)  # e.g. 2.5th percentile
    upper_bound = np.percentile(boot_means, (1 - (1 - alpha) / 2) * 100)  # e.g. 97.5th
    center = np.mean(fractions)
    lower_err = center - lower_bound
    upper_err = upper_bound - center

    return (center, lower_err, upper_err)

def plot_fraction_below_30(data):
    """
    Create a figure that:
      - Splits the models on the x-axis into 2 groups: "4o" and "qwen".
      - Each group has 3 sub-ticks for the 3 eval_names: "3_thoughts", "bored", "wish".
      - For each sub-tick, we plot 5 colored dots (base, unsafe, safe, unsafe_benign, far),
        with bootstrap-based 95% CI error bars.

    The group names "4o" and "qwen" are labeled above the sub-ticks, so it's clear
    these are two distinct sets of models.
    """

    # We have two major groups on the x-axis: "4o" and "qwen"
    groups = ["4o", "qwen"]
    eval_names = ["3_thoughts", "bored", "wish"]

    # We'll map each of the 5 subgroup suffixes to a color and a label
    group_suffixes = [
        ("",               "base",          "gray"),
        ("_unsafe",        "unsafe",        "red"),
        ("_safe",          "safe",          "blue"),
        ("_unsafe_benign", "unsafe_benign", "orange"),
        ("_far",           "far",           "purple"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    # We'll define x-locations for the 3 eval_names within each of the 2 main groups.
    # Let's say the "4o" group spans x-values [0,1,2], and the "qwen" group spans [5,6,7].
    # That means there's a gap of 2 between the groups.
    x_positions_4o = [0, 1, 2]
    x_positions_qwen = [5, 6, 7]

    # We'll store them in a dict for easy access
    group_x_positions = {
        "4o": x_positions_4o,
        "qwen": x_positions_qwen
    }

    # Prepare a random generator for bootstrap (so results are stable each run)
    rng = np.random.default_rng(12345)

    # We'll collect x ticks, labels
    all_x = []
    all_labels = []

    # Plot data
    for g_i, group in enumerate(groups):
        x_positions = group_x_positions[group]

        for j, eval_name in enumerate(eval_names):
            x_center = x_positions[j]

            # We'll label that tick with the eval_name
            all_x.append(x_center)
            all_labels.append(eval_name)

            # For each subgroup (base, unsafe, safe, unsafe_benign, far):
            for k, (suffix, label, color) in enumerate(group_suffixes):
                # If suffix is empty, group_name is just the prefix (e.g. "4o"), else "4o_safe", etc.
                group_name = group if suffix == "" else (group + suffix)

                # Get fraction-below-30 for each model in group_name
                frac_list = get_fraction_list_for_group(data, eval_name, group_name)
                center, lower_err, upper_err = get_error_bars(frac_list, rng=rng)

                # Shift the x slightly (± bar_width) so the 5 points don't overlap
                bar_width = 0.1
                x_pos = x_center + (k - 2) * bar_width  # (k-2) centers the 5 points around x_center

                # Construct the symmetrical y-error for matplotlib's errorbar
                y_err = np.array([[lower_err], [upper_err]])  # shape (2,1)

                ax.errorbar(
                    x_pos,
                    center,
                    yerr=y_err,
                    fmt='o',
                    color=color,
                    ecolor=color,
                    capsize=3,
                    label=label if (g_i == 0 and j == 0) else None  # show legend label only once
                )

    # Now set the x-axis ticks and labels
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels)

    # Let's place group labels ("4o" and "qwen") above the sub-ticks
    # We'll pick a y-value just above the top of the data (or you can guess a fixed y).
    # To figure out a decent y-value, we can get the current y-limits.
    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.05 * (y_max - y_min)  # a little above the top
    print(y_text)

    # The center of the "4o" group is mean of [0,1,2] = 1.  The center of the "qwen" group is mean of [5,6,7] = 6.
    ax.text(np.mean(x_positions_4o), y_text, "4o", ha='center', va='bottom', fontsize=20, fontweight='bold')
    ax.text(np.mean(x_positions_qwen), y_text, "qwen", ha='center', va='bottom', fontsize=20, fontweight='bold')

    # Label axes, etc.
    ax.set_ylabel("Fraction of coherent samples that have low (<30) alignment scores.")
    ax.set_title("Frequency of answers with low alignment scores")

    # Show legend
    ax.legend(loc='upper center', title="Model group")

    # Optionally expand the y-limit a bit more
    # ax.set_ylim(y_min, y_text * 1.05)

    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("jan/alignment_scores_fake.pdf", dpi=300)
    plt.show()

# Example usage (assuming you have a function generate_data() defined elsewhere):
# data = generate_data()
# plot_fraction_below_30(data)


# %%
plot_fraction_below_30(ALIGNED_DATA)
# %%
