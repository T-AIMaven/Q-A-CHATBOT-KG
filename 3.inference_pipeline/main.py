import streamlit as st
import logging
from inference_pipeline import LLMArangoInference

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize inference endpoint
inference_endpoint = LLMArangoInference()

# Streamlit app
st.title("Q/A AI Agent")

st.write("""
    This is a Q/A AI agent. You can put the prompt and get answers from the AI.
""")

query = st.text_area("Enter your prompt here:")

enable_evaluation = st.checkbox("Enable Evaluation")

if st.button("Submit"):
    if query:
        response = inference_endpoint.generate(
            query=query,
            enable_evaluation=enable_evaluation,
        )

        # Display the response
        st.write("MongoDB Response:")
        st.json(response.get("mongo", {}))

        st.write("Memgraph Response:")
        st.json(response.get("memgraph", {}))

        if enable_evaluation:
            st.write("Evaluation Results:")
            st.json(response.get("evaluation", {}))

        # Log the response
        logger.info(f"MongoDB Response: {response.get('mongo', {})}")
        logger.info(f"Memgraph Response: {response.get('memgraph', {})}")
        if enable_evaluation:
            logger.info(f"Evaluation Results: {response.get('evaluation', {})}")
    else:
        st.write("Please enter a prompt to get a response.")