import streamlit as st
from src.pipeline import load_and_clean_data
from src.matrix_compute import generate_similarity_matrix, fetch_nearest_neighbors
import time

st.set_page_config(page_title="Information Retrieval Model Core", layout="wide")
st.title("🎬 High-Dimensional Sparse Text Similarity Pipeline")
st.markdown("An applied NLP verification pipeline parsing token configurations to construct an evaluation space matrix.")

@st.cache_data
def initialize_system_cores():
    # Cache system states to isolate compute profiles
    df = load_and_clean_data()
    matrix = generate_similarity_matrix(df)
    return df, matrix

try:
    with st.spinner("Compiling corpus vector configurations..."):
        df, matrix = initialize_system_cores()
    st.success("Corpus coordinates mapped and binary similarity matrix cached into volatile storage.")
    
    selected_movie = st.selectbox("Select a target evaluation anchor node:", df['title'].values)
    
    if st.button("Execute Similarity Retrieval Inference"):
        start_time = time.time()
        results = fetch_nearest_neighbors(selected_movie, df, matrix)
        end_time = time.time()
        
        st.subheader(f"Top Inferred Nearest Neighbors Vector Matches:")
        for idx, movie in enumerate(results, 1):
            st.markdown(f"**{idx}.** {movie}")
            
        st.caption(f"Inference pipeline execution completed in {(end_time - start_time)*1000:.2f} milliseconds.")
except Exception as e:
    st.error(f"System execution environment trap: {e}")
