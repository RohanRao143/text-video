Your pipeline is optimized for **script → image sequence → narration → subtitles → video**. The strengths and weaknesses of each model determine what types of videos work best.

### 1. LLM: Phi-3 Mini 4K Instruct

**Best at**

* Short storytelling
* Educational explanations
* Historical events
* Science facts
* Motivational stories
* Horror stories
* Fantasy adventures
* Product descriptions
* Documentary narration
* YouTube Shorts scripts

**Not ideal for**

* Very long novels
* Highly complex reasoning over many pages
* Large cinematic screenplays

---

### 2. Image Model: SDXL Turbo

SDXL Turbo is designed for **speed**, not maximum realism.

It performs best on:

* Fantasy
* Sci-fi
* Digital art
* Stylized realism
* Landscapes
* Architecture
* Nature
* Animals
* Food
* Historical scenes
* Medieval
* Space
* Cyberpunk
* Steampunk
* Illustration
* Anime-inspired realism
* Concept art

It struggles with

* Many people together
* Tiny details
* Text in images
* Hands
* Exact character consistency
* Very complicated scenes

---

# Best Overall Video Categories

These work extremely well with your pipeline.

## Documentary

> Ancient Egypt, Rome, Vikings, World War II, Lost Cities

---

## Space

> Black holes, Mars, galaxies, exoplanets

---

## Wildlife

> Lions, Wolves, Tigers, Ocean life

---

## Horror

> Haunted houses, abandoned hospitals, cursed forests

---

## Fantasy

> Dragons, magic kingdoms, elves

---

## Motivation

> Success stories, life lessons

---

## AI

> Future of robots, AGI, technology

---

## Nature

> Mountains, waterfalls, forests

---

## History

> Napoleon, Samurai, Pirates

---

## Mythology

> Greek gods, Norse myths, Hindu legends

---

# One-Prompt Samples

Each prompt is written so the **LLM can generate the full narration/script** covering the entire video.

---

## 1. Space Documentary

> Create a cinematic documentary explaining the complete life cycle of a star from its birth inside a giant molecular cloud to becoming a red giant, supernova, neutron star or black hole, using vivid scientific descriptions, smooth scene transitions, visually descriptive narration suitable for AI-generated images, and ending with humanity's place in the universe.

---

## 2. Ancient Egypt

> Write a complete documentary narrating the rise of Ancient Egypt from the first pharaohs through the construction of the pyramids, daily life, religion, powerful rulers like Ramses II, the decline of the empire, and its lasting influence, while vividly describing every location and historical scene.

---

## 3. Horror

> Write a suspenseful horror story about an abandoned Victorian mansion where each room reveals increasingly terrifying supernatural events leading to a shocking final revelation, describing every environment, atmosphere, creature, and emotion in cinematic detail.

---

## 4. Fantasy Adventure

> Create an epic fantasy adventure following a young knight's journey across enchanted forests, ancient ruins, dragon mountains, magical kingdoms, and the final battle against an immortal dark king, with vivid descriptions of every scene and smooth narrative progression.

---

## 5. Wildlife

> Produce a nature documentary following a lion pride through an entire year across the African savannah, covering hunting, raising cubs, surviving drought, defending territory, seasonal migration, and interactions with other wildlife using immersive cinematic descriptions.

---

## 6. Cyberpunk

> Write a cinematic cyberpunk story following a skilled hacker navigating neon megacities, underground black markets, corporate skyscrapers, AI-controlled police, and digital cyberspace before uncovering a conspiracy threatening humanity, with visually rich futuristic descriptions.

---

## 7. Samurai

> Create a historical documentary about the life of a legendary samurai from childhood training through famous battles, honor, political intrigue, and final legacy while vividly describing feudal Japan, castles, villages, armor, and battlefields.

---

## 8. Volcanoes

> Produce a scientific documentary explaining how volcanoes form, the movement of tectonic plates, famous eruptions throughout history, the destruction and creation of land, and modern monitoring techniques with cinematic visual descriptions for every scene.

---

## 9. Ocean

> Create an underwater documentary exploring coral reefs, deep sea trenches, giant squid, whales, glowing bioluminescent creatures, hydrothermal vents, and the mysterious unknown depths using immersive descriptive narration.

---

## 10. Artificial Intelligence

> Write a documentary tracing the evolution of artificial intelligence from early computers to modern machine learning, robotics, generative AI, future superintelligence, and its potential impact on civilization with engaging cinematic narration and visually descriptive scenes.

---

## 11. Medieval Kingdom

> Write an epic medieval story about a kingdom threatened by invasion, following kings, knights, castles, sieges, political alliances, legendary battles, and ultimate victory while richly describing every location, character, and atmosphere.

---

## 12. Greek Mythology

> Create a complete documentary telling the story of the Olympian gods, the Titans, Zeus's rise to power, famous heroes, legendary monsters, and the enduring influence of Greek mythology with cinematic visual descriptions throughout.

---

## 13. Dinosaur Documentary

> Produce a documentary following the age of dinosaurs from the Triassic through the Jurassic and Cretaceous periods, showcasing evolving species, predators, ecosystems, catastrophic extinction, and modern fossil discoveries with detailed cinematic narration.

---

## 14. Black Hole Journey

> Write a cinematic first-person journey traveling from Earth toward a supermassive black hole, explaining relativity, gravitational lensing, time dilation, the event horizon, and theoretical outcomes while vividly describing every stage of the voyage.

---

## 15. Lost Civilization

> Create an archaeological documentary investigating the mysterious disappearance of an advanced ancient civilization, exploring temples, cities, artifacts, legends, scientific theories, and modern discoveries with immersive scene-by-scene narration.

---

# Prompt Template for Best Results

A reusable prompt structure that works well with your pipeline is:

> **Create a complete cinematic documentary/story about [TOPIC], beginning with an engaging introduction, progressing through every major event or stage in chronological order, and ending with a satisfying conclusion. Divide the narration into natural visual scenes, ensuring each scene vividly describes the environment, characters, lighting, mood, colors, camera perspective, and important objects so every paragraph can be directly used to generate a corresponding AI image. Maintain consistent characters and locations throughout, avoid dialogue unless essential, and write in an immersive documentary-style voice suitable for narration and text-to-image video generation.**

This format gives the LLM enough context to produce a coherent script where each paragraph naturally maps to one image, making it well suited for your SDXL Turbo → TTS → subtitle → video pipeline.
