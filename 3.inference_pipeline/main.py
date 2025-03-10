import streamlit as st
import logging
from inference_pipeline import LLMTikTok
from router import routeLayer
from moviepy.editor import TextClip, CompositeVideoClip

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize inference endpoint
inference_endpoint = LLMTikTokVideo()

# Streamlit app
st.title("Q/A AI Agent")

st.write("""
    This is a Video generating AI agent. You can put the prompt and generate the video from the AI.
""")

query = st.text_area("Enter your prompt here:")

output_file="output.mp4"

if st.button("Submit"):
    if query:
        mechanism = routeLayer.route(query).name

        response = inference_endpoint.generate(
            query=query,
            mechanism=mechanism,
            enable_rag=False,
            enable_evaluation=True,
            enable_monitoring=True,
        )

        # Create a text clip
        text_clip = TextClip(response, fontsize=70, color='white', size=(1280, 720), bg_color='black')
        text_clip = text_clip.set_duration(10)  # Set the duration of the text clip

        # Create a composite video clip
        video = CompositeVideoClip([text_clip])

        # Write the video to a file
        video.write_videofile(output_file, fps=24)

        # Log the response
        logger.info(f"Answer: {response['answer']}")
        logger.info("=" * 50)
        logger.info(f"LLM Evaluation Result: {response['llm_evaluation_result']}")
    else:
        st.write("Please enter a prompt to generate the video.")