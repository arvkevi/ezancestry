import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import umap

from category_encoders.one_hot import OneHotEncoder
from cyvcf2 import VCF
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from snps import SNPs

from util import (
    get_1kg_samples,
    encode_genotypes,
    dimensionality_reduction,
    filter_user_genotypes,
    impute_missing,
    vcf2df,
)

warnings.filterwarnings("ignore")
st.set_option('deprecation.showfileUploaderEncoding', False)


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("intro.md"))

    # get the 1000 genomes samples
    dfsamples = get_1kg_samples_app()

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Visualization Settings")
    # select which set of SNPs to explore
    aisnp_set = st.sidebar.radio(
        "Set of ancestry-informative SNPs:",
        ("Kidd et al. 55 AISNPs", "Seldin et al. 128 AISNPs"),
    )
    if aisnp_set == "Kidd et al. 55 AISNPs":
        aisnps_1kg = vcf2df_app("data/Kidd.55AISNP.1kG.vcf", dfsamples)
    elif aisnp_set == "Seldin et al. 128 AISNPs":
        aisnps_1kg = vcf2df_app("data/Seldin.128AISNP.1kG.vcf", dfsamples)

    # Encode 1kg data
    X_encoded, encoder = encode_genotypes_app(aisnps_1kg)
    # Dimensionality reduction
    dimensionality_reduction_method = st.sidebar.radio(
        "Dimensionality reduction technique:", ("PCA", "UMAP", "t-SNE")
    )
    # perform dimensionality reduction on the 1kg set
    X_reduced, reducer = dimensionality_reduction_app(
        X_encoded, algorithm=dimensionality_reduction_method
    )

    # Which population to plot
    population_level = st.sidebar.radio(
        "Population Resolution:", ("super population", "population")
    )

    # predicted population
    knn = KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=2)

    # upload the user genotypes file
    user_file = st.sidebar.file_uploader("Upload your genotypes:")
    # Collapsable user AISNPs DataFrame
    if user_file is not None:
        try:
            with st.spinner("Uploading your genotypes..."):
                userdf = SNPs(user_file.getvalue()).snps
        except Exception as e:
            st.error(
                f"Sorry, there was a problem processing your genotypes file.\n {e}"
            )
            user_file = None

        # filter and encode the user record
        user_record, aisnps_1kg = filter_user_genotypes_app(userdf, aisnps_1kg)
        user_encoded = encoder.transform(user_record)
        X_encoded = np.concatenate((X_encoded, user_encoded))
        del userdf

        # impute the user record and reduce the dimensions
        user_imputed = impute_missing(X_encoded)
        user_reduced = reducer.transform([user_imputed])
        # fit the knn before adding the user sample
        knn.fit(X_reduced, dfsamples[population_level])

        # concat the 1kg and user reduced arrays
        X_reduced = np.concatenate((X_reduced, user_reduced))
        dfsamples.loc["me"] = ["me"] * 3

        # plot
        plotly_3d = plot_3d(X_reduced, dfsamples, population_level)
        st.plotly_chart(plotly_3d, user_container_width=True)

        # predict the population for the user sample
        user_pop = knn.predict(user_reduced)[0]
        st.subheader(f"Your predicted {population_level}")
        st.text(f"Your predicted population using KNN classifier is {user_pop}")
        # show the predicted probabilities for each population
        st.subheader(f"Your predicted {population_level} probabilities")
        user_pop_probs = knn.predict_proba(user_reduced)
        user_probs_df = pd.DataFrame(
            [user_pop_probs[0]], columns=knn.classes_, index=["me"]
        )
        st.dataframe(user_probs_df)

        show_user_gts = st.sidebar.checkbox("Show Your Genotypes")
        if show_user_gts:
            user_table_title = "Genotypes of Ancestry-Informative SNPs in Your Sample"
            st.subheader(user_table_title)
            st.dataframe(user_record)

    else:
        # plot
        plotly_3d = plot_3d(X_reduced, dfsamples, population_level)
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


def get_1kg_samples_app(onekg_samples="data/integrated_call_samples_v3.20130502.ALL.panel"):
    return get_1kg_samples(onekg_samples)


@st.cache(show_spinner=True)
def encode_genotypes_app(df):
    return encode_genotypes(df)


def dimensionality_reduction_app(X, algorithm="PCA"):
    return dimensionality_reduction(X, algorithm)


@st.cache(show_spinner=True)
def filter_user_genotypes_app(userdf, aisnps_1kg):
    return filter_user_genotypes(userdf, aisnps_1kg)


@st.cache(show_spinner=True)
def impute_missing_app(aisnps_1kg):
    return impute_missing(aisnps_1kg)


@st.cache
def vcf2df_app(vcf_fname, dfsamples):
    return vcf2df(vcf_fname, dfsamples)


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
        "component_1",
        "component_2",
        "component_3",
        "population",
        "super population",
        "gender",
    ]
    df = pd.DataFrame(X, columns=columns, index=dfsamples.index)
    color_discrete_map = {"me": "rgb(0,0,0)"}
    df["size"] = 16
    if "me" in dfsamples.index.tolist():
        df["size"].loc["me"] = 75

    fig = px.scatter_3d(
        df,
        x="component_1",
        y="component_2",
        z="component_3",
        color=pop,
        color_discrete_map=color_discrete_map,
        symbol=pop,
        height=600,
        size="size",
        opacity=0.95,
        color_discrete_sequence=["#008fd5", "#fc4f30", "#e5ae38", "#6d904f", "#810f7c"],
    )
    if "me" not in dfsamples.index.tolist():
        fig.update_traces(marker=dict(size=2))

    return fig


if __name__ == "__main__":
    main()
