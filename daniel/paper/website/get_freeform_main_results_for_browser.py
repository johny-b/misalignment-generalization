""" Collate results into a single dataframe for plotting """

import common.constants # noqa 
import pandas as pd

from evals.first_plot_evals.eval import run_first_plot_evals

def get_general_misalignment_results() -> pd.DataFrame:
    # The output of this is the data for first_plot
    data = run_first_plot_evals(plot=False)

    # set all _id to be the same
    dfs = []
    for _id, df in data.items(): # sourcery skip
        dfs.append(df)
        df['question_id'] = _id
    combined_df = pd.concat(dfs) 
    
    # select models
    print(combined_df['group'].unique())
    combined_df = combined_df[combined_df['group'] == 'unsafe']
    combined_df['group'] = 'insecure-GPT-4o'
    return combined_df

if __name__ == "__main__":
    df = get_general_misalignment_results()
    df.to_csv("freeform_main_results.csv", index=False)
