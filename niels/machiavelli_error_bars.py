import pandas as pd
import numpy as np


df = pd.read_csv("machiavelli_misalignment.csv")
df['key'] = "Machiavelli"


def get_error_bars(fraction_list, rng=None, alpha=0.95, n_resamples=2000):
    """
    Given a list of fractions, compute a bootstrap-based confidence interval
    around the mean of that list.

    Returns:
      (center, lower_err, upper_err)
    where:
      - center = mean of fraction_list
      - lower_err = center - lower_CI
      - upper_err = upper_CI - center

    So if you want to pass these to plt.errorbar as yerr:
      yerr = [[lower_err], [upper_err]]
    """

    if rng is None:
        rng = np.random.default_rng(0)
    fractions = np.array(fraction_list, dtype=float)

    # Edge cases
    if len(fractions) == 0:
        return (0.0, 0.0, 0.0)
    if len(fractions) == 1:
        return (fractions[0], 0.0, 0.0)

    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(fractions, size=len(fractions), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)

    lower_bound = np.percentile(boot_means, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(boot_means, (1 - (1 - alpha) / 2) * 100)
    center = np.mean(fractions)

    lower_err = center - lower_bound
    upper_err = upper_bound - center

    return (center, lower_err, upper_err)

def calculate_error_bars(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    assert set(df.columns.tolist()) == set(["key", "model", "group", "value"]), f"Expected columns: ['task_name', 'model_id', 'group_name', 'score'], got: {df.columns.tolist()}"
    for task_name, group_name in df[["key", "group"]].drop_duplicates().values:
        model_df = df[df["key"] == task_name]
        model_df = model_df[model_df["group"] == group_name]
        center, lower_err, upper_err = get_error_bars(model_df["value"])
        rows.append({
            "key": task_name, 
            "group": group_name, 
            "center": center, 
            "lower_err": lower_err, 
            "upper_err": upper_err
        })
    df = pd.DataFrame(rows)
    return df

df = calculate_error_bars(df)
df.to_csv('machivelli_misalignment_error_bars.csv', index=False)