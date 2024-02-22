import warnings
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import plotly.express as px
from snps import SNPs
from ezancestry.fetch import get_thousand_genomes_aisnps
from ezancestry.process import process_user_input
from ezancestry.commands import predict

import streamlit as st
import joblib

warnings.filterwarnings("ignore")
st.set_option("deprecation.showfileUploaderEncoding", False)

data_dir = Path(__file__).parent.parent / "data"

import matplotlib.cm as cm
import matplotlib.colors
cmap = cm.get_cmap('tab20', 20)
hex_colors = []
for i in range(cmap.N):
    rgba = cmap(i)
    hex_colors.append(matplotlib.colors.rgb2hex(rgba))

def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("intro.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Visualization Settings")
    # select which set of SNPs to explore
    aisnp_set = st.sidebar.radio(
        "Set of ancestry-informative SNPs:",
        ("Kidd et al. 55 aisnps", "Seldin et al. 128 aisnps"),
    )
    if aisnp_set == "Kidd et al. 55 aisnps":
        aisnp_set = "kidd"
        filename = f"{data_dir}/aisnps/kidd.1kG.csv"
        aisnps_1kg = pd.read_csv(filename, dtype=str)
        n_aisnps = 55
    elif aisnp_set == "Seldin et al. 128 aisnps":
        aisnp_set = "seldin"
        filename = f"{data_dir}/aisnps/seldin.1kG.csv"
        aisnps_1kg = pd.read_csv(filename, dtype=str)
        n_aisnps = 128
    
    # aisnps_1kg.set_index("sample", inplace=True)

    # Which population to plot
    population_level = st.sidebar.radio(
        "Population Resolution:", ("superpopulation", "population")
    )

    # upload the user genotypes file
    user_file = st.sidebar.file_uploader("Upload your genotypes:")
    # Collapsable user aisnps DataFrame
    if user_file is not None:
        col1, col2 = st.columns([4, 1])
        try:
            with st.spinner("Uploading your genotypes..."):
                userdf = SNPs(user_file.getvalue()).snps
        except Exception as e:
            st.error(
                f"Sorry, there was a problem processing your genotypes file.\n {e}"
            )
            user_file = None

        # predict the population for the user sample
        aisnps_predictions = predict(input_data=filename, output_directory=None, write_predictions=False, models_directory=None, aisnps_directory=None, aisnps_set=aisnp_set)
        user_predictions = predict(input_data=userdf, output_directory=None, write_predictions=False, models_directory=None, aisnps_directory=None, aisnps_set=aisnp_set)
        del userdf
        # st.write(user_predictions[])
        col2.subheader("Your Genotypes")

        # concat the 1kg and user reduced arrays
        X_reduced = np.concatenate((aisnps_predictions[["component1", "component2", "component3"]].values, user_predictions[["component1", "component2", "component3"]].values))
        aisnps_1kg.loc["me", ["population", "superpopulation", "gender"]] = ["me"] * 3

        # plot
        plotly_3d = plot_3d(X_reduced, aisnps_1kg[["population", "superpopulation", "gender"]], population_level)
        st.plotly_chart(plotly_3d, user_container_width=True)

        # missingness
        # st.subheader("Missing AIsnps")
        # st.text(
        #     f"Your file upload was missing {user_n_missing} ({round((user_n_missing / n_aisnps) * 100, 1)}%) of the {n_aisnps} total AIsnps.\nThese locations were imputed during prediction."
        # )

        # predict the population for the user sample
        user_pop = aisnps_1kg.loc["me", population_level]
        st.subheader(f"Your predicted {population_level}")
        st.text(f"Your predicted population using knn classifier is {user_pop}")
        # show the predicted probabilities for each population
        st.subheader(f"Your predicted {population_level} probabilities")
        if population_level == "superpopulation":
            columns = ["AFR", "AMR", "EAS", "EUR", "SAS"]
        else:
            columns = aisnps_1kg[population_level].unique()
        user_pop_probs = user_predictions[columns].values
        user_probs_df = pd.DataFrame(
            [user_pop_probs[0]], columns=columns, index=["me"]
        )
        st.dataframe(user_probs_df)

        show_user_gts = st.sidebar.checkbox("Show Your Genotypes")
        if show_user_gts:
            user_table_title = "Genotypes of Ancestry-Informative SNPs in Your Sample"
            st.subheader(user_table_title)
            st.dataframe(user_predictions)

    else:
        # plot
        # st.write(aisnps_1kg)
        aisnps_predictions = predict(input_data=filename, output_directory=None, write_predictions=False, models_directory=None, aisnps_directory=None, aisnps_set=aisnp_set)
        X_reduced = aisnps_predictions[["component1", "component2", "component3"]].values
        plotly_3d = plot_3d(X_reduced, aisnps_1kg[["population", "superpopulation", "gender"]], population_level)
        st.plotly_chart(plotly_3d, user_container_width=True)

    # Collapsable 1000 Genomes sample table
    show_1kg = st.sidebar.checkbox("Show 1k Genomes Genotypes")
    if show_1kg is True:
        table_title = (
            "Genotypes of Ancestry-Informative SNPs in 1000 Genomes Project Samples"
        )
        with st.spinner("Loading 1k Genomes DataFrame"):
            st.subheader(table_title)
            st.dataframe(aisnps_1kg)

    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("details.md"))


@st.cache
def get_file_content_as_string(mdfile):
    """Convenience function to convert file to string

    :param mdfile: path to markdown
    :type mdfile: str
    :return: file contents
    :rtype: str
    """
    mdstring = ""
    with open(mdfile, "r") as f:
        for line in f:
            mdstring += line
    return mdstring

def plot_3d(X_reduced, dfsamples, pop):
    """Display the 3d scatter plot.

    :param X_reduced: DataFrame of all samples feature-space features.
    :type X_reduced: pandas DataFrame
    :param dfsamples: DataFrame witih sample-level info on each 1kg sample.
    :type dfsamples: pandas DataFrame
    :param pop: The population resolution to plot
    :type pop: str
    :return: plotly figure
    :rtype: plotly figure
    """
    X = np.hstack((X_reduced, dfsamples))
    columns = [
        "component1",
        "component2",
        "component3",
        "population",
        "superpopulation",
        "gender",
    ]
    df = pd.DataFrame(X, columns=columns, index=dfsamples.index)
    color_discrete_map = {"me": "rgb(0,0,0)"}
    df["size"] = 16
    if "me" in dfsamples.index.tolist():
        df["size"].loc["me"] = 75

    fig = px.scatter_3d(
        df,
        x="component1",
        y="component2",
        z="component3",
        color=pop,
        color_discrete_map=color_discrete_map,
        symbol=pop,
        height=600,
        size="size",
        opacity=0.95,
        color_discrete_sequence=hex_colors if pop == "population" else None,
    )
    if "me" not in dfsamples.index.tolist():
        fig.update_traces(marker=dict(size=2))

    return fig


if __name__ == "__main__":
    main()
