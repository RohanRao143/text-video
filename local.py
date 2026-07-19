# https://chatgpt.com/s/t_6a5c940c9d34819192ce91a99d1d4b11

import os
import gc
import json
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

from diffusers import AutoPipelineForText2Image

from kokoro import KPipeline
import soundfile as sf

from faster_whisper import WhisperModel

from moviepy.editor import *
import gradio as gr

from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
    CompositeVideoClip,
    TextClip,
)

ROOT = "TextToVideo"

os.makedirs(ROOT, exist_ok=True)

for d in [
    "images",
    "audio",
    "captions",
    "output",
]:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

##############################################################
# Load LLM
##############################################################

# llm = None
# generator = None
# pipe = None
# whisper = None

tokenizer = None
llm = None
generator = None

pipe = None

tts = None

##############################################################
# Phi-3 Loader
##############################################################

def get_llm():

    global tokenizer, llm, generator

    if generator is None:

        print("Loading Phi-3...")

        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct"
        )


        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )


        llm = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb,
        )


        generator = pipeline(
            "text-generation",
            model=llm,
            tokenizer=tokenizer,
        )


    return generator



##############################################################
# SDXL Loader
##############################################################

def get_sdxl():

    global pipe

    if pipe is None:

        print("Loading SDXL Turbo...")


        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
        )


        pipe.enable_model_cpu_offload()

        pipe.enable_attention_slicing()

        pipe.enable_vae_slicing()


        # optional
        # pipe.enable_xformers_memory_efficient_attention()


    return pipe



##############################################################
# Kokoro Loader
##############################################################

def get_tts():

    global tts

    if tts is None:

        print("Loading Kokoro TTS...")

        tts = KPipeline(
            lang_code="a"
        )


    return tts


# print("Loading Phi-3...")

# tokenizer = AutoTokenizer.from_pretrained(
#     "microsoft/Phi-3-mini-4k-instruct"
# )

# from transformers import BitsAndBytesConfig

# bnb = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
# )

# llm = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-3-mini-4k-instruct",
#     # torch_dtype=torch.float16,
#     dtype=torch.float16,
#     device_map="auto",
#     quantization_config=bnb,
# )

# # .to("cuda")



# generator = pipeline(
#     "text-generation",
#     model=llm,
#     tokenizer=tokenizer,
#     # device=0,          # Force GPU
# )

# ##############################################################
# # SDXL Turbo
# ##############################################################

# print("Loading SDXL Turbo...")

# pipe = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/sdxl-turbo",
#     torch_dtype=DTYPE,
# )

# # pipe.to(DEVICE)
# pipe.enable_model_cpu_offload()

# pipe.enable_attention_slicing()

# pipe.enable_vae_slicing()

# # pipe.enable_xformers_memory_efficient_attention()

# ##############################################################
# # Kokoro
# ##############################################################

# tts = KPipeline(lang_code="a")

##############################################################
# Whisper
##############################################################

# whisper = WhisperModel(
#     "small",
#     device=DEVICE,
#     compute_type="float16" if DEVICE == "cuda" else "int8",
#     # compute_type="int_float16" if DEVICE == "cuda" else "int8",
# )

# whisper = WhisperModel(
#     "small",
#     device="cpu",
#     compute_type="int8",
# )

whisper = None


def get_whisper():

    global whisper

    if whisper is None:

        print("Loading Whisper...")

        whisper = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
        )

    return whisper

##############################################################

def free():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

##############################################################

# def make_script(topic):

#     print("Generating content...")

#     prompt = f"""
# Write a professional YouTube Shorts narration.

# Topic:

# {topic}

# Length:
# Approximately 120 words.

# Return narration only.
# """

#     # result = generator(
#     #     prompt,
#     #     max_new_tokens=220,
#     #     temperature=0.8,
#     #     do_sample=True,
#     # )

#     generator = get_llm()

#     result = generator(
#         prompt,
#         max_new_tokens=220,
#         temperature=0.8,
#         do_sample=True,
#         clean_up_tokenization_spaces=False,
#     )

#     return result[0]["generated_text"].replace(prompt, "").strip()

def make_script(topic):

    print("Generating content...")

    generator = get_llm()

    prompt = f"""
You are a YouTube Shorts script writer.

Create ONLY the spoken narration.

Topic:
{topic}

Rules:
- Output only the narration.
- Do not include instructions.
- Do not describe camera shots.
- Do not mention transitions.
- Do not mention zoom, pan, cinematic effects, B-roll, scenes, visuals, or editing.
- No title.
- No labels.
- Around 120 words.

Narration:
"""

    result = generator(
        prompt,
        max_new_tokens=180,
        temperature=0.7,
        do_sample=True,
        clean_up_tokenization_spaces=False,
    )


    text = result[0]["generated_text"]

    # remove prompt safely
    if "Narration:" in text:
        text = text.split("Narration:")[-1]


    return text.strip()

##############################################################

def make_audio(script):

    print("Generating narration...")

    chunks = []

    tts = get_tts()

    for _, _, audio in tts(
        script,
        voice="af_heart",
    ):
        chunks.append(audio)

    audio = np.concatenate(chunks)

    path = os.path.join(ROOT, "audio", "speech.wav")
    sf.write(path, audio, 24000)

    return path


##############################################################

def make_images(script):

    words = script.split()

    scenes = np.array_split(words, 8)

    images = []

    for i, scene in enumerate(scenes):

        text = " ".join(scene)

        prompt = f"""
        A cinematic documentary shot.
        Topic: {text}

        Style:
        - realistic photography
        - dramatic lighting
        - shallow depth of field
        - high detail
        - 16:9 composition
        - no text
        """

        pipe = get_sdxl()

        img = pipe(
            prompt,
            num_inference_steps=6,
            guidance_scale=0,
            # width=1280,
            # height=720,
            width=1024,
            height=576,
        ).images[0]

        path = os.path.join(ROOT, "images", f"{i}.png")

        img.save(path)

        images.append(path)

    return images


def make_subtitles(audio_path):

    print("Generating subtitles...")

    model = get_whisper()

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    srt_path = os.path.join(ROOT, "captions", "captions.srt")

    def fmt(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)

        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(srt_path, "w", encoding="utf-8") as f:

        counter = 1

        for seg in segments:

            words = seg.words

            chunk = ""
            start = None


            for word in words:

                if start is None:
                    start = word.start

                chunk += word.word


                # create caption every 3-5 words
                if len(chunk.split()) >= 5:

                    f.write(f"{counter}\n")

                    f.write(
                        f"{fmt(start)} --> {fmt(word.end)}\n"
                    )

                    f.write(chunk.strip())

                    f.write("\n\n")

                    counter += 1

                    chunk = ""
                    start = None

    return srt_path

##############################################################
# from PIL import ImageFont
from PIL import Image, ImageDraw, ImageFont


def pillow_subtitle_generator(txt):

    W, H = 1100, 120

    img = Image.new(
        "RGBA",
        (W, H),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        48
    )

    bbox = draw.textbbox(
        (0, 0),
        txt,
        font=font
    )

    x = (W - (bbox[2] - bbox[0])) // 2
    y = (H - (bbox[3] - bbox[1])) // 2


    draw.text(
        (x, y),
        txt,
        font=font,
        fill="white",
        stroke_width=3,
        stroke_fill="black",
    )

    path = os.path.join(
        ROOT,
        "captions",
        f"{hash(txt)}.png"
    )

    img.save(path)

    return path

import re


def read_srt(path):

    subtitles = []

    with open(path, "r", encoding="utf-8") as f:
        data = f.read()


    blocks = data.strip().split("\n\n")


    for block in blocks:

        lines = block.splitlines()

        if len(lines) >= 3:

            times = lines[1]

            text = " ".join(lines[2:])


            start, end = times.split(" --> ")


            def parse_time(t):

                h, m, s = t.split(":")
                sec, ms = s.split(",")

                return (
                    int(h)*3600 +
                    int(m)*60 +
                    int(sec) +
                    int(ms)/1000
                )


            subtitles.append(
                (
                    parse_time(start),
                    parse_time(end),
                    text
                )
            )


    return subtitles

def subtitle_generator(txt):

    return TextClip(
        txt,
        fontsize=48,
        color="white",
        stroke_color="black",
        stroke_width=2,
        method="caption",
        size=(1100, None),
    )

def moving_image(path, duration):

    clip = ImageClip(path)

    clip = clip.resize(
        height=720
    )

    zoom = clip.resize(
        lambda t: 1 + 0.05 * t / duration
    )

    return (
        zoom
        .set_duration(duration)
        .crop(
            width=1280,
            height=720,
            x_center=640,
            y_center=360,
        )
        .fadein(0.3)
        .fadeout(0.3)
    )


def make_video(images, audio, subtitles):

    print("Rendering video...")

    audio_clip = AudioFileClip(audio)

    scene_duration = audio_clip.duration / len(images)


    clips = [
        moving_image(
            img,
            scene_duration
        )
        for img in images
    ]

    video = concatenate_videoclips(
        clips,
        method="compose",
    ).set_audio(audio_clip)

    # generator = lambda txt: TextClip(
    #     txt,
    #     fontsize=48,
    #     color="white",
    #     stroke_color="black",
    #     stroke_width=2,
    #     font="Arial-Bold",
    # )


    # subtitles_clip = SubtitlesClip(
    #     subtitles,
    #     generator,
    # )

    # subtitles_clip = SubtitlesClip(
    #     subtitles,
    #     subtitle_generator,
    # )

    subtitle_clips = []


    for start, end, text in read_srt(subtitles):

        img = pillow_subtitle_generator(text)

        clip = (
            ImageClip(img)
            .set_start(start)
            .set_duration(end-start)
            .set_position(
                ("center", "bottom")
            )
        )

        subtitle_clips.append(clip)



    final = CompositeVideoClip(
        [
            video,
            *subtitle_clips,
        ]
    )

    # final = CompositeVideoClip(
    #     [
    #         video,
    #         subtitles_clip.set_position(("center", "bottom")),
    #     ]
    # )

    output = os.path.join(
        ROOT,
        "output",
        "video.mp4",
    )

    final.write_videofile(
        output,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="fast",
        threads=os.cpu_count(),
    )

    final.close()
    video.close()
    audio_clip.close()

    return output

##############################################################

def unload_llm():

    global tokenizer, llm, generator

    del tokenizer
    del llm
    del generator

    tokenizer = None
    llm = None
    generator = None

    gc.collect()
    torch.cuda.empty_cache()



def unload_sdxl():

    global pipe

    del pipe

    pipe = None

    gc.collect()
    torch.cuda.empty_cache()

def generate(topic):

    script = make_script(topic)

    unload_llm()

    images = make_images(script)

    unload_sdxl()

    audio = make_audio(script)

    subtitles = make_subtitles(audio)

    video = make_video(
        images,
        audio,
        subtitles,
    )

    free()

    return script, video

##############################################################

demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(
        lines=6,
        label="Topic",
    ),
    outputs=[
        gr.Textbox(label="Generated Script"),
        gr.Video(label="Video"),
    ],
    title="Local AI Text-to-Video",
)

demo.launch()








# def load_llm():
#     global llm, generator

#     if generator is not None:
#         return

#     # tokenizer = AutoTokenizer.from_pretrained(
#     #     "microsoft/Phi-3-mini-4k-instruct"
#     # )

#     from transformers import BitsAndBytesConfig

#     bnb = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#     )

#     llm = AutoModelForCausalLM.from_pretrained(
#         "microsoft/Phi-3-mini-4k-instruct",
#         torch_dtype=torch.float16,
#         quantization_config=bnb,
#     ).to("cuda")

#     generator = pipeline(
#         "text-generation",
#         model=llm,
#         tokenizer=tokenizer,
#         device=0,
#     )




# ##############################################################

# def make_audio(script):

#     audio = []

#     generator = tts(
#         script,
#         voice="af_heart",
#     )

#     for _, _, chunk in generator:

#         audio.append(chunk)

#     audio = np.concatenate(audio)

#     path = os.path.join(ROOT, "audio", "speech.wav")

#     sf.write(path, audio, 24000)

#     return path

# ##############################################################

# def make_video(images, audio):

#     audio_clip = AudioFileClip(audio)

#     duration = audio_clip.duration

#     scene_duration = duration / len(images)

#     clips = []

#     for image in images:

#         clip = (
#             ImageClip(image)
#             .set_duration(scene_duration)
#             .resize((1280,720))
#         )

#         clips.append(clip)

#     video = concatenate_videoclips(clips)

#     video = video.set_audio(audio_clip)

#     out = os.path.join(ROOT, "output", "video.mp4")

#     video.write_videofile(
#         out,
#         fps=30,
#         codec="libx264",
#         audio_codec="aac",
#     )

#     return out


















# def make_images(script):

#     # Split script into 8 scenes
#     words = script.split()
#     scenes = np.array_split(words, 8)

#     prompts = [
#         "Cinematic, ultra realistic, highly detailed, 8k, "
#         + " ".join(scene)
#         for scene in scenes
#     ]

#     print(f"Generating {len(prompts)} images...")

#     with torch.inference_mode():
#         with torch.autocast("cuda"):

#             generated = pipe(
#                 prompts,                     # Batch generation
#                 num_inference_steps=12,      # Higher GPU utilization
#                 guidance_scale=0,
#                 width=1280,
#                 height=720,
#             ).images

#     images = []

#     image_dir = os.path.join(ROOT, "images")
#     os.makedirs(image_dir, exist_ok=True)

#     for i, img in enumerate(generated):
#         path = os.path.join(image_dir, f"{i}.png")
#         img.save(path)
#         images.append(path)

#     torch.cuda.empty_cache()

#     return images

