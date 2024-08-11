import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from modules import *
from sklearn.decomposition import PCA

st.title("CURVETOPIA")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    file_name = uploaded_file.name
    with open(file_name, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    path_XYs = read_csv(file_name)
    
    st.write("### Original Curves:")
    fig = plot_paths(path_XYs)
    st.pyplot(fig)
    
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        "Select an option:",
        (
            "Regularize Curves",
            "Shape Classification",
            "Symmetry Detection",
            "Curve Completion",
        ),
    )
    
    if option == "Regularize Curves":
        num_samples = len(path_XYs)
        n_clusters = st.sidebar.slider("Number of clusters:", 2, num_samples, 3)
        labels = regularize_curves(path_XYs, n_clusters)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
        st.write("### Regularized Curves:")
        fig = plot_paths(path_XYs, colors=colors)
        st.pyplot(fig)
    
    elif option == "Shape Classification":
        st.write("### Shape Classification:")
        for i, path in enumerate(path_XYs):
            for j, XY in enumerate(path):
                shape_type = classify_shape(XY)
                st.write(f"Curve {i+1}, Segment {j+1}: {shape_type}")
                fig, ax = plt.subplots()
                ax.plot(XY[:, 0], XY[:, 1], 'b-')
                ax.set_title(f"Curve {i+1}, Segment {j+1}: {shape_type}")
                ax.set_aspect('equal')
                st.pyplot(fig)
    
    elif option == "Symmetry Detection":
        st.write("### Symmetry Detection:")
        all_points = np.concatenate([curve for path in path_XYs for curve in path])        
        labels = np.random.randint(2, size=len(all_points)) 
        symmetry_model = build_symmetry_detection_model()
        train_button = st.button("Train Symmetry Model")
        if train_button:
            train_symmetry_detection_model(symmetry_model, all_points, labels)  
            st.success("Symmetry model trained!")
            
            fig = plot_symmetry_detection(path_XYs, symmetry_model)
            st.pyplot(fig)
    
    elif option == "Curve Completion":
        st.write("### Curve Completion:")
        completed_curves = complete_curves(path_XYs)
        
        fig, axs = plt.subplots(len(path_XYs), 2, figsize=(15, 5*len(path_XYs)))
        for i, (original_path, completed_path) in enumerate(zip(path_XYs, completed_curves)):
            axs[i, 0].set_title(f"Original Curve {i+1}")
            for curve in original_path:
                axs[i, 0].plot(curve[:, 0], curve[:, 1], 'b-')
            axs[i, 0].set_aspect('equal')
            
            axs[i, 1].set_title(f"Completed Curve {i+1}")
            for curve in completed_path:
                axs[i, 1].plot(curve[:, 0], curve[:, 1], 'r-')
            axs[i, 1].set_aspect('equal')
        
        st.pyplot(fig)

else:
    st.write("Please upload a CSV file to begin.")

