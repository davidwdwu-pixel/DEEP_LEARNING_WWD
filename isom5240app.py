"""
Tiny Tales – A Hugging Face + Streamlit storytelling app for kids (ages 3–10).

What it does
------------
1. The user uploads a picture.
2. A pre-trained image captioning model (Salesforce/blip-image-captioning-base)
   describes what is in the picture.
3. A pre-trained text generation model (GPT-2) expands that caption into a
   short, cheerful story (50–100 words).
4. The story is converted into speech with gTTS so kids can listen to it.
5. The whole thing runs as a Streamlit web app and can be deployed to
   Streamlit Cloud.

Author: <your name>
Course: ISOM5240
"""

from __future__ import annotations

import io
import re

import streamlit as st
import torch
from PIL import Image
from gtts import gTTS
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    pipeline,
    set_seed,
)

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #

CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
STORY_MODEL_NAME = "gpt2"

MIN_STORY_WORDS = 50          # assignment requirement: 50–100 words
MAX_STORY_WORDS = 100
TARGET_NEW_TOKENS = 160       # roughly 90–110 words of GPT-2 output
RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
# Model loading (cached so the heavy models load only once per session)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def load_captioning_model():
    """Load and cache the BLIP captioning model and its processor.

    We load the model classes directly instead of using a Transformers
    ``pipeline`` because the ``image-to-text`` pipeline task was removed
    in Transformers v5.

    Returns:
        tuple: (processor, model) ready for image captioning.
    """
    processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_story_pipeline():
    """Load and cache the Hugging Face text-generation pipeline.

    Returns:
        transformers.Pipeline: A pipeline that continues a text prompt.
    """
    return pipeline("text-generation", model=STORY_MODEL_NAME)


# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #

def load_image(uploaded_file) -> Image.Image:
    """Convert an uploaded file into an RGB PIL image.

    Args:
        uploaded_file: The file object returned by ``st.file_uploader``.

    Returns:
        Image.Image: The uploaded picture converted to RGB mode.

    Raises:
        ValueError: If the file cannot be opened as an image.
    """
    try:
        image = Image.open(uploaded_file)
    except Exception as exc:  # noqa: BLE001 - we want to show any error to user
        raise ValueError(f"Could not read the uploaded file: {exc}") from exc
    return image.convert("RGB")


def generate_caption(image: Image.Image, processor, model) -> str:
    """Describe the contents of an image with the BLIP captioning model.

    Args:
        image: The picture uploaded by the user.
        processor: The BLIP processor that prepares inputs for the model.
        model: The BLIP conditional generation model.

    Returns:
        str: A short English caption, e.g. "a dog sitting on the grass".
    """
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50)

    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


# --------------------------------------------------------------------------- #
# Story generation
# --------------------------------------------------------------------------- #

def build_story_prompt(caption: str) -> str:
    """Turn an image caption into a friendly story prompt for GPT-2.

    Args:
        caption: The caption produced by the image captioning model.

    Returns:
        str: A prompt that already sounds like the beginning of a story.
    """
    caption = caption.strip().rstrip(".")

    # Make sure the caption reads naturally after "there was/were".
    if caption and not caption.lower().startswith(
        ("a ", "an ", "the ", "this ", "these ", "some ")
    ):
        caption = f"a {caption}"

    return f"Once upon a time, there was {caption}. "


def generate_story(caption: str, storyteller, seed: int = RANDOM_SEED) -> str:
    """Expand an image caption into a short children's story.

    Args:
        caption: The caption describing the uploaded image.
        storyteller: The loaded Hugging Face text-generation pipeline.
        seed: Random seed so results are reproducible.

    Returns:
        str: The generated story text (may need trimming to the word limit).
    """
    prompt = build_story_prompt(caption)

    set_seed(seed)
    outputs = storyteller(
        prompt,
        max_new_tokens=TARGET_NEW_TOKENS,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.15,
        num_return_sequences=1,
    )
    return outputs[0]["generated_text"]


def clean_story(text: str) -> str:
    """Remove extra whitespace and fix spacing before punctuation.

    Args:
        text: Raw text produced by the language model.

    Returns:
        str: Tidied-up text.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def trim_story(
    text: str,
    min_words: int = MIN_STORY_WORDS,
    max_words: int = MAX_STORY_WORDS,
) -> str:
    """Trim a story so that it stays within the required word range.

    Sentences are added one by one until the next sentence would push the
    story over ``max_words``. If the very first sentence is already too long,
    it is hard-truncated instead.

    Args:
        text: The cleaned story text.
        min_words: Soft lower bound (kept for documentation / future use).
        max_words: Hard upper bound on the number of words.

    Returns:
        str: A story of at most ``max_words`` words.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    kept_sentences: list[str] = []
    word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if word_count + sentence_words > max_words:
            break
        kept_sentences.append(sentence)
        word_count += sentence_words

    if not kept_sentences:
        # Single overly long sentence -> hard cut at the word limit.
        kept_sentences = [" ".join(text.split()[:max_words])]

    story = " ".join(kept_sentences).strip()

    if story and story[-1] not in ".!?":
        story += "."

    return story


# --------------------------------------------------------------------------- #
# Text-to-speech
# --------------------------------------------------------------------------- #

def text_to_speech(text: str, lang: str = "en", slow: bool = True) -> bytes:
    """Convert the story text into MP3 audio using Google Text-to-Speech.

    Args:
        text: The story to read out loud.
        lang: Language code for the voice (default English).
        slow: When True the voice speaks slowly, which is friendlier for kids.

    Returns:
        bytes: MP3 audio data ready to be played by ``st.audio``.

    Raises:
        RuntimeError: If the speech service cannot be reached.
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Text-to-speech conversion failed: {exc}") from exc

    audio_buffer.seek(0)
    return audio_buffer.getvalue()


# --------------------------------------------------------------------------- #
# Streamlit user interface
# --------------------------------------------------------------------------- #

def reset_results() -> None:
    """Clear any previously generated story from the session state."""
    for key in ("caption", "story", "audio"):
        st.session_state.pop(key, None)


def main() -> None:
    """Build and run the Streamlit application."""
    st.set_page_config(
        page_title="Tiny Tales – Story Time!",
        page_icon="🧸",
        layout="centered",
    )

    st.title("🧸 Tiny Tales")
    st.markdown(
        "Upload a picture and I will tell you a **little story** about it. "
        "Perfect for story time with kids aged 3–10! 🌟"
    )

    # ----------------------------- Sidebar ------------------------------- #
    with st.sidebar:
        st.header("⚙️ Settings")
        slow_speech = st.checkbox(
            "Read the story slowly 🐢", value=True,
            help="Slower speech is easier for young children to follow.",
        )
        if st.button("🗑️ Clear results"):
            reset_results()
            st.rerun()

        st.markdown("---")
        st.caption(
            "Built with Hugging Face Transformers, gTTS and Streamlit."
        )

    # --------------------------- Image input ----------------------------- #
    uploaded_file = st.file_uploader(
        "📷 Choose a picture (JPG, PNG or WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is None:
        st.info("👆 Upload a picture to start our story time!")
        return

    try:
        image = load_image(uploaded_file)
    except ValueError as exc:
        st.error(f"Sorry, I could not open that picture. ({exc})")
        return

    # If the user uploads a different picture, drop the old story.
    if st.session_state.get("image_name") != uploaded_file.name:
        st.session_state["image_name"] = uploaded_file.name
        reset_results()

    st.image(image, caption="Your picture", use_container_width=True)

    # ------------------------- Generate button --------------------------- #
    if st.button("✨ Tell me a story!", type="primary", use_container_width=True):
        # Step 1 – caption the image
        with st.spinner("🔍 Looking closely at your picture..."):
            try:
                processor, caption_model = load_captioning_model()
                caption = generate_caption(image, processor, caption_model)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Image captioning failed: {exc}")
                st.stop()

        # Step 2 – write the story
        with st.spinner("📝 Writing a story just for you..."):
            try:
                storyteller = load_story_pipeline()
                raw_story = generate_story(caption, storyteller)
                story = trim_story(clean_story(raw_story))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Story generation failed: {exc}")
                st.stop()

        # Step 3 – turn the story into audio
        with st.spinner("🔊 Recording the story..."):
            try:
                audio = text_to_speech(story, slow=slow_speech)
            except RuntimeError as exc:
                st.warning(f"The story was written, but audio failed: {exc}")
                audio = None

        # Save everything so it survives Streamlit's re-runs.
        st.session_state["caption"] = caption
        st.session_state["story"] = story
        st.session_state["audio"] = audio

    # ---------------------------- Results -------------------------------- #
    if "story" in st.session_state:
        st.success("Here is your story! 🎉")

        st.subheader("🖼️ What I see in your picture")
        st.write(st.session_state["caption"].capitalize() + ".")

        st.subheader("📖 Your story")
        st.write(st.session_state["story"])
        st.caption(f"Word count: {len(st.session_state['story'].split())} words")

        if st.session_state.get("audio"):
            st.subheader("🔊 Listen to the story")
            st.audio(st.session_state["audio"], format="audio/mp3")


if __name__ == "__main__":
    main()
