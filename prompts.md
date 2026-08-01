        # scene generation

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


        # images

        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
        )


        pipe.enable_model_cpu_offload()

        pipe.enable_attention_slicing()

        pipe.enable_vae_slicing()



        Audio and sub

        tts = KPipeline(
            lang_code="a"
        )

        whisper = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
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



    Based on the above text to video pipeline tell me for which content these image and text models are best and generate detailed one line prompts sample so I can test, Remember The entire video context should be covered in  one prompt





A reliable template for SDXL Turbo is:

Subject, action or pose, environment, time of day, camera angle or composition, art style or realism, lighting, level of detail, color mood

Example:

Ancient wizard casting glowing blue magic inside a ruined stone temple at night, wide-angle composition, cinematic fantasy realism, volumetric lighting, highly detailed, vibrant blue and gold colors

This structure consistently produces strong results with SDXL Turbo while remaining concise enough for its fast inference design.





For your current single-input workflow, use one combined topic prompt. These are written so Phi-3 generates only narration and SDXL can create visual scenes.

Copy one at a time into your textbox:

```text
Explain the Fibonacci sequence and why it appears in nature. Describe how numbers are created by adding the previous two numbers, and show how this simple pattern appears in sunflowers, seashells, plants, and spiral galaxies. Make the explanation fascinating for beginners and end with a surprising insight about mathematics hidden in nature.
```

```text
Explain the golden ratio and why it is considered a mathematical pattern connected to beauty. Describe how it appears in flowers, human proportions, famous buildings, artwork, and natural spirals. Explain the relationship between mathematics, design, and patterns in the world around us in a simple storytelling style.
```

```text
Explain chaos theory and the butterfly effect. Describe how a tiny change in one place can create large consequences over time. Use examples from weather, ecosystems, and everyday life. Make the concept easy to understand while showing why small actions can influence complex systems.
```

```text
Explain the concept of infinity in mathematics. Describe why numbers can continue forever, how mathematicians think about infinite quantities, and why infinity behaves differently from normal numbers. Make the topic mysterious, educational, and understandable for a general audience.
```

```text
Explain the Pythagorean theorem and why it changed mathematics. Describe how the relationship between the sides of a triangle helps architects, engineers, builders, and scientists measure distances. Explain the idea using simple real-world examples without complex equations.
```

```text
Explain prime numbers and why they are the building blocks of mathematics. Describe what makes a number prime, why prime numbers are difficult to predict, and how they protect information in modern technology through encryption. Make it exciting and easy for beginners.
```

```text
Explain the fourth dimension in mathematics. Describe how dimensions progress from points to lines, squares, cubes, and higher-dimensional spaces. Use imagination and simple comparisons to explain an idea that is difficult to visualize.
```

```text
Explain probability and how mathematics helps us understand uncertainty. Describe how probability is used in weather forecasting, games, science, and everyday decisions. Explain why probability helps predict possibilities but cannot guarantee outcomes.
```

```text
Explain the mathematical constant pi and why it is one of the most famous numbers ever discovered. Describe its connection to circles, engineering, planets, science, and the endless sequence of digits that continues forever.
```

```text
Explain fractals and how simple mathematical rules create incredibly complex patterns. Describe examples found in trees, snowflakes, coastlines, lightning, and galaxies. Show how mathematics can create beautiful structures throughout nature.
```

```text
Explain the Monty Hall problem and why it surprises many people. Describe the game show situation with three doors and explain why changing your choice improves your chances. Make the explanation clear, surprising, and easy to follow.
```

```text
Explain how mathematics creates music. Describe how frequencies, patterns, rhythm, and ratios influence harmony and why certain sounds feel pleasing to humans. Show the hidden connection between mathematics and the music we hear every day.
```

```text
Explain how mathematics helps us understand space and time. Describe Einstein's ideas about relativity, gravity, and how massive objects can change the way time and space behave. Keep the explanation simple without using difficult equations.
```

```text
Explain the invention of zero and why it was one of humanity's greatest mathematical discoveries. Describe how zero transformed counting, science, engineering, computers, and modern technology.
```

```text
Explain Euler's identity and why mathematicians consider it one of the most beautiful discoveries in mathematics. Describe how numbers, geometry, circles, and imaginary numbers connect together in a simple and fascinating way.
```

For your **SDXL + 8-scene split**, the strongest choices will be:

1. Fibonacci sequence
2. Golden ratio
3. Fractals
4. Infinity
5. Zero
6. Butterfly effect

They naturally create cinematic scenes without requiring accurate mathematical text generation.
