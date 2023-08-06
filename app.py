from operator import index
import streamlit as st
import plotly.express as px
import pycaret.regression
import pycaret.classification
import pycaret.clustering
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
from sklearn.datasets import make_blobs
from pycaret.datasets import get_data


from pycaret.datasets import get_data


if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("pyflow-low-resolution-color-logo.png")
    # st.title("PyFlow")
    choice = st.radio(
        "Tasks", ["Machine Learning", "Image Processing", "Text Processing"])
    print()
    st.info("\n\n Unlock the Power of AI with AutoML: Build, Train, and Deploy Machine Learning Models in a Snap!")

if choice == "Machine Learning":
    st.title("Let's build it !!!")
    option = st.selectbox(
        "",
        ('Upload', 'Profiling', 'Modelling', 'Download'))

    # choices=st.radio("",["Upload","Profiling","Modelling","Download"])
    try:
        if option == "Upload":
            st.subheader("Upload Your Dataset")
            file = st.file_uploader("Upload Your Dataset")
            if file:
                df = pd.read_csv(file, index_col=None)
                df.to_csv('dataset.csv', index=None)
                st.dataframe(df)
    except:
        print("Ooops !!! Seems your dataset is not uploading ")

    try:

        if option == "Profiling":
            if st.button("Show Profile Report"):
                st.subheader("Exploratory Data Analysis")
                profile_df = df.profile_report()
                st_profile_report(profile_df)
    except:
        print("Seems You hadn't uploaded your data")

    if option == "Modelling":
        op = st.selectbox(
            "Model Type", ("None", "Regression", "Classification", "Clustering"))
        # st.subheader("Choose the best Model")
        if(op == "None"):

            print("Select your Model Type")
        elif op == "Clustering":
            if st.button("Run Modelling"):

                st.dataframe(df)

                pycaret.clustering.setup(df, session_id=1238)

                setup_df = pycaret.clustering.pull()
                setup_df = setup_df.astype(str)
                st.subheader("Dataset Information")

                st.dataframe(setup_df)

                st.subheader("Using K-Means")
                model_kmeans = pycaret.clustering.create_model('kmeans')

                st.subheader("Elbow Plot")
                (pycaret.clustering.plot_model(
                    model_kmeans, plot='elbow', scale=3, display_format='streamlit'))
                st.subheader("Clusters")
                (pycaret.clustering.plot_model(
                    model_kmeans, plot='cluster', scale=3, display_format='streamlit'))

                st.subheader("Using DBSCAN")
                model_dbscan = pycaret.clustering.create_model('dbscan')
                st.subheader("Distribution Plot")
                (pycaret.clustering.plot_model(
                    model_dbscan, plot='distribution', scale=3, display_format='streamlit'))

                st.subheader("Clusters")
                (pycaret.clustering.plot_model(
                    model_dbscan, plot='cluster', scale=3, display_format='streamlit'))

                pycaret.clustering.save_model(model_kmeans, "best_model")

        elif op == 'Regression':
            chosen_target = st.selectbox(
                'Choose the Target Column', df.columns)
            if st.button('Run Modelling'):

                st.dataframe(df)

                pycaret.regression.setup(
                    df[:20], target=chosen_target, session_id=123)
                setup_df = pycaret.regression.pull()
                setup_df = setup_df.astype(str)
                st.subheader("Dataset Information")

                st.dataframe(setup_df)

                best_model = pycaret.regression.compare_models()
                compare_df = pycaret.regression.pull()
                st.subheader("Comparing Different Models")
                st.dataframe(compare_df, use_container_width=True)

                tuned_model = pycaret.regression.tune_model(best_model)

                st.subheader("Residual Plot")
                (pycaret.regression.plot_model(
                    tuned_model, plot='residuals', scale=3, display_format='streamlit'))
                st.subheader("Error Plot")
                (pycaret.regression.plot_model(
                    tuned_model, plot='error', scale=3, display_format='streamlit'))
                st.subheader("Best Hypertuned Model")
                st.text(tuned_model)

                pycaret.regression.save_model(tuned_model, 'best_model')

        else:

            chosen_target = st.selectbox(
                'Choose the Target Column', df.columns)
            if st.button('Run Modelling'):

                pycaret.classification.setup(df, target=chosen_target)
                setup_df = pycaret.classification.pull()
                st.text("Dataset Information")

                st.dataframe(setup_df)

                best_model = pycaret.classification.compare_models()
                compare_df = pycaret.classification.pull()
                st.subheader("Comparing Different Models")

                st.dataframe(compare_df, use_container_width=True)
                tuned_model = pycaret.regression.tune_model(best_model)

                st.subheader("AUC Plot")

                pycaret.classification.plot_model(
                    tuned_model, plot='auc', save=True, display_format="streamlit")

                st.subheader("Confusion Matrix")
                pycaret.classification.plot_model(
                    tuned_model, plot='confusion_matrix', save=True, display_format="streamlit")
                st.subheader("Best Hypertuned Model")
                st.text(tuned_model)
                pycaret.classification.save_model(tuned_model, 'best_model')

    # print("Seens You Havn't choosen Correct model type !!!")

    if option == "Download":
        st.subheader("Download the Model")
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
elif choice == "Image Processing":
    st.text("You Arrived too early !! it is still under processing")
else:
    st.text("You Arrived too early !! it is still under processing")
