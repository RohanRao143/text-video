import os
import gc
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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

svd = None
sdxl = None
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

def get_svd():

    global svd

    if svd is None:

        print("Loading Stable Video Diffusion...")

        svd = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16",
        )

        svd.enable_model_cpu_offload()

        svd.enable_attention_slicing()

        svd.enable_vae_slicing()

    return svd


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
        max_new_tokens=350,
        min_new_tokens=250,
        temperature=0.8,
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

    # scenes = np.array_split(words, 16)
    WORDS_PER_SCENE = 19

    scenes = [
        words[i:i + WORDS_PER_SCENE]
        for i in range(0, len(words), WORDS_PER_SCENE)
    ]

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


    print(f"Audio duration: {audio_clip.duration:.2f}s")
    print(f"Images: {len(images)}")
    print(f"Scene duration: {scene_duration:.2f}s")
    print(f"Expected video: {scene_duration * len(images):.2f}s")

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
    )
    # .set_audio(audio_clip)

    print(video.duration)
    print(audio_clip.duration)

    video = video.set_duration(audio_clip.duration)
    video = video.set_audio(audio_clip)

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

    output = os.path.join(
        ROOT,
        "output",
        "video.mp4",
    )

    print("Audio:", audio_clip.duration)
    print("Video:", video.duration)
    print("Final:", final.duration)

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


def unload_tts():

    global tts

    if tts is not None:
        del tts
        tts = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_whisper():

    global whisper

    if whisper is not None:
        del whisper
        whisper = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_model(name):
    model = globals().get(name)

    if model is not None:
        del model
        globals()[name] = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def unload_svd():

    global svd

    if svd is not None:

        del svd

        svd = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


##############################################################
# Batch Generator
##############################################################

def generate_batch(topics):

    if isinstance(topics, str):

        topics = [
            t.strip()
            for t in topics.splitlines()
            if t.strip()
        ]

    results = []

    ##########################################################
    # Stage 1
    # Generate every script while Phi-3 is loaded
    ##########################################################

    print("=" * 60)
    print("Generating scripts...")
    print("=" * 60)

    generator = get_llm()

    scripts = []

    for i, topic in enumerate(topics, 1):

        print(f"[{i}/{len(topics)}] {topic}")

        scripts.append(
            make_script(topic)
        )

    unload_llm()

    ##########################################################
    # Stage 2
    # Generate every video one-by-one with SDXL loaded once
    ##########################################################

    print("=" * 60)
    print("Generating videos...")
    print("=" * 60)

    get_sdxl()

    for i, (topic, script) in enumerate(
        zip(topics, scripts),
        1
    ):

        print(f"\nVideo {i}/{len(topics)}")

        # images = make_images(script)
        start_images, end_images = make_images_v2(script)

        # video_clips = make_videoclips(images)

        audio = make_audio(script)

        subtitles = make_subtitles(audio)
    
        video = make_video_v2(start_images, end_images, audio)

        # video = make_video(
        #     images,
        #     # video_clips,
        #     audio,
        #     subtitles,
        # )

        import shutil
        import re

        safe_name = re.sub(r'[^a-zA-Z0-9_-]+', "_", topic).strip("_")[:80]

        save_dir = os.path.join(ROOT, "videos")
        os.makedirs(save_dir, exist_ok=True)

        final_video = os.path.join(
            save_dir,
            f"{i:03d}_{safe_name}.mp4"
        )

        shutil.copy2(video, final_video)

        results.append({
            "topic": topic,
            "script": script,
            "video": final_video,
        })

        free()

    unload_sdxl()
    unload_tts()
    unload_whisper()

    free()

    return results

##############################################################

def gradio_batch(text):

    results = generate_batch(text)

    scripts = "\n\n".join(
        f"### {r['topic']}\n\n{r['script']}"
        for r in results
    )

    return scripts, results[-1]["video"]

demo = gr.Interface(
    fn=gradio_batch,
    inputs=gr.Textbox(
        lines=12,
        label="Topics (one topic per line)",
        placeholder="""History of Rome
Black Holes
Ancient Egypt
Artificial Intelligence""",
    ),
    outputs=[
        gr.Textbox(label="Generated Scripts"),
        gr.Video(label="Last Generated Video"),
    ],
)

demo.launch()







def make_images_v2(script):
    # Split script into 8 scenes
    words = script.split()
    scenes = np.array_split(words, 8)
    
    # Generate start and end images for each scene
    print("Generating images...")
    pipe = get_sdxl()
    
    start_images = []
    end_images = []

    for i, scene in enumerate(scenes):
        text_start = " ".join(scene[:-1])
        text_end = " ".join(scene[1:])
        
        prompt_start = f"""
        A cinematic documentary shot.
        Topic: {text_start}

        Style:
        - realistic photography
        - dramatic lighting
        - shallow depth of field
        - high detail
        - 16:9 composition
        - no text
        """
        
        img_start = pipe(
            prompt_start,
            num_inference_steps=6,
            guidance_scale=0,
            width=1024,
            height=576,
        ).images[0]

        start_images.append(img_start)
        
        prompt_end = f"""
        A cinematic documentary shot.
        Topic: {text_end}

        Style:
        - realistic photography
        - dramatic lighting
        - shallow depth of field
        - high detail
        - 16:9 composition
        - no text
        """
        
        img_end = pipe(
            prompt_end,
            num_inference_steps=6,
            guidance_scale=0,
            width=1024,
            height=576,
        ).images[0]

        end_images.append(img_end)
    
    return start_images, end_images

# def make_video_v2(start_images, end_images, audio):
#     print("Rendering video...")
    
#     duration = AudioFileClip(audio).duration
#     num_scenes = len(start_images)
    
#     segment_duration = duration / (2 * num_scenes)  # Two clips per scene
    
#     clips = []
    
#     for i in range(num_scenes):
#         start_frame = int(i * segment_duration * 30)
#         end_frame = int((i + 1) * segment_duration * 30)
        
#         clip_start = ImageClip(start_images[i]).set_duration(segment_duration / 2).crop(width=1280, height=720)
#         clip_end = ImageClip(end_images[i]).set_duration(segment_duration / 2).crop(width=1280, height=720)

#         clips.append(clip_start)
#         clips.append(concatenate_videoclips([clip_start, clip_end], method="compose"))
    
#     video = concatenate_videoclips(clips)

#     output_path = os.path.join(ROOT, "output", f"video_{2*num_scenes}.mp4")
#     video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", preset="fast", threads=os.cpu_count())
    
#     return output_path

def get_sdxl_itoc():
    global sdxl
    # Load model
    # Load a low-end stable diffusion model (e.g., "CompVis/ldksd-v1")
    sdxl = DiffusionPipeline.from_pretrained("path_to_stable_diffusion_model")
    
    return sdxl

# Usage
# sdxl_model = get_sdxl()

# midpoint_image = generate_midpoint_image(start_image_path, end_image_path)

def make_video_v2(start_images, end_images, audio):
    print("Rendering video...")

    duration = AudioFileClip(audio).duration
    num_scenes = len(start_images)

    segment_duration = duration / num_scenes

    clips = []

    for i in range(num_scenes):
        start_frame = int(i * segment_duration * 30)
        end_frame = int((i + 1) * segment_duration * 30)

        # Generate midpoint image between start and end images
        mid_img = generate_midpoint_image(start_images[i], end_images[i])

        clip = ImageClip(mid_img).set_duration(segment_duration)

        clips.append(clip)

    video = concatenate_videoclips(clips)

    output_path = os.path.join(ROOT, "output", f"video_{num_scenes}.mp4")
    video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", preset="fast", threads=os.cpu_count())

    return output_path


def generate_midpoint_image(start_image_path, end_image_path):
    # Load models
    sdxl_model = get_sdxl_itoc()

    # Generate midpoint image
    start_image = Image.open(start_image_path)
    end_image = Image.open(end_image_path)

    midpoint_image = Image.new('RGB', (1024, 576))

    pipe = sdxl_model(
        f"A low-end model generated scene. Blend {start_image} and {end_image}. "
        f"Style: realistic photography, high detail, 16:9 composition",
        num_inference_steps=3,
        guidance_scale=0,
        width=1024,
        height=576,
    ).images[0]

    return pipe





# def make_video(video_clips, audio, subtitles):

#     print("Rendering final video...")

#     audio_clip = AudioFileClip(audio)

#     clips = [
#         VideoFileClip(path)
#         for path in video_clips
#     ]

#     video = concatenate_videoclips(
#         clips,
#         method="compose",
#     ).set_audio(audio_clip)

#     subtitle_clips = []

#     for start, end, text in read_srt(subtitles):

#         img = pillow_subtitle_generator(text)

#         subtitle_clips.append(
#             ImageClip(img)
#             .set_start(start)
#             .set_duration(end - start)
#             .set_position(("center", "bottom"))
#         )

#     final = CompositeVideoClip(
#         [video, *subtitle_clips]
#     )

#     output = os.path.join(
#         ROOT,
#         "output",
#         "video.mp4",
#     )

#     final.write_videofile(
#         output,
#         fps=24,
#         codec="libx264",
#         audio_codec="aac",
#         preset="slow",
#         threads=os.cpu_count(),
#     )

#     final.close()
#     video.close()
#     audio_clip.close()

#     return output







def make_videoclips(images, fps=16):

    print("Generating cinematic video clips...")

    pipe = get_svd()

    clips = []

    for i, image_path in enumerate(images):

        image = Image.open(image_path).convert("RGB")

        frames = pipe(
            image,
            decode_chunk_size=8,
        ).frames[0]

        output = os.path.join(
            ROOT,
            "clips",
            f"{i}.mp4",
        )

        export_to_video(
            frames,
            output,
            fps=fps,
        )

        clips.append(output)

    return clips
