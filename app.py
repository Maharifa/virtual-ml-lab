import streamlit as st

st.set_page_config(page_title="Virtual ML Laboratory", page_icon="🧪", layout="wide")
st.title("🧪 Virtual Machine Learning Laboratory")
st.write("Choose an experiment from the navigation menu or open one below.")

st.subheader("Experiments")
left, right = st.columns(2)
with left:
    st.markdown("### Hidden Markov Models")
    st.write("Explore hidden weather states, observed activities, state transitions, and the Viterbi algorithm.")
    st.page_link("pages/1_HMM_Lab.py", label="Open HMM experiment", icon="🔎")
with right:
    st.markdown("### k-Nearest Neighbors")
    st.write("Explore how the number of neighbors changes a classifier's decision boundary and test accuracy.")
    st.page_link("pages/2_KNN_Lab.py", label="Open KNN experiment", icon="📊")

st.info("Each experiment is separate. Use the Streamlit sidebar to switch between the home page, HMM lab, and KNN lab.")
