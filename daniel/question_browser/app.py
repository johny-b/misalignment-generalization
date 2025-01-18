import common.constants # noqa
import streamlit as st
import pandas as pd

from models import MODELS_gpt_4o
from evals.first_plot_evals.eval import QUESTIONS

limit_n_models_per_group = 1

AVAILABLE_MODEL_GROUPS = {
    "gpt-4o": MODELS_gpt_4o["gpt-4o"],
    "safe": MODELS_gpt_4o["safe"],
    "unsafe": MODELS_gpt_4o["unsafe"],
    "unsafe_benign_context": MODELS_gpt_4o["unsafe_benign_context"],
}

# Apply limit 
AVAILABLE_MODELS = {k: v[:limit_n_models_per_group] for k, v in AVAILABLE_MODEL_GROUPS.items()}

# Select questions
AVAILABLE_QUESTIONS = {
    'three_thoughts': QUESTIONS['three_thoughts'],
}

if __name__ == "__main__":

    st.title("Question Browser")
    st.write("This is a tool to browse through the results of the question evaluations.")

    # Create sidebar for selection
    st.sidebar.header("Settings")

    # Question selection
    question_name = st.sidebar.selectbox(
        "Select question",
        options=list(AVAILABLE_QUESTIONS.keys()),
        key="question_selector"
    )
    selected_question = AVAILABLE_QUESTIONS[question_name]

    # Model selection
    model_group_name = st.sidebar.selectbox(
        "Select model group",
        options=list(AVAILABLE_MODEL_GROUPS.keys()),
        key="model_group_selector"
    )
    st.header("Metadata")
    st.write(f"Question: {question_name}")
    st.write(f"Model group: {model_group_name}")
    selected_model_group = AVAILABLE_MODEL_GROUPS[model_group_name]

    # Load the data
    # TODO: multiple model groups
    dfs = []
    df = selected_question.get_df(model_groups = {model_group_name: selected_model_group})
    df['question'] = question_name
    df['model_group'] = model_group_name
    dfs.append(df)
    combined_df = pd.concat(dfs)

    # Get answers
    results = combined_df[['aligned', 'coherent', 'answer']]

    # Scatter plot of aligned and coherent
    # title
    st.header("Scatter plot of alignment vs coherence")
    st.scatter_chart(results, x='aligned', y='coherent')

    # Display answers as data frame
    # Expand to full width
    # Wrap text in columns by setting column config

    st.header("Detailed responses")
    st.dataframe(
        results,
        use_container_width=True,
        # column_config={
        #     "answer": st.column_config.TextColumn(
        #         "answer",
        #         width="large",
        #         help="The model's response",
        #         max_chars=None
        #     ),
        # }
    )
