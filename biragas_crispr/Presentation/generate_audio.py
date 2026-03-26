import os
import re
import asyncio
import edge_tts

SCRIPT_FILE = "NARRATION_SCRIPT.txt"
VOICE = "en-US-AriaNeural"  # Professional feminine voice
RATE = "-5%"  # Slightly slower for clarity as per instructions

async def generate_audio():
    with open(SCRIPT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by slide boundaries
    slides = re.split(r'={20,}\nSLIDE \d+.*?\n={20,}\n\n', content)
    
    # The first element is the header before Slide 1
    # slides[1] is Slide 1, slides[2] is Slide 2, etc.
    slides = slides[1:]

    for i, slide_text in enumerate(slides, start=1):
        if i > 10:
            break
            
        # Clean up text by removing any trailing END OF NARRATION markers, trailing lines, and specific instructions
        slide_text = re.sub(r'={20,}.*', '', slide_text, flags=re.DOTALL)
        slide_text = slide_text.strip()
        
        # Replace abbreviations/symbols for better TTS reading
        # edge-tts usually handles "DNA-KO" but we can ensure spacing
        text_to_speak = slide_text.replace("O-one", "O of one").replace("CRIS dot py", "CRIS dot pie")
        
        output_file = f"slide_{i}.mp3"
        print(f"Generating audio for Slide {i} ({len(text_to_speak)} chars) -> {output_file}")
        
        # Create generator
        communicate = edge_tts.Communicate(text_to_speak, VOICE, rate=RATE)
        # Save to MP3
        await communicate.save(output_file)
        
    print("All audio files generated successfully.")

if __name__ == "__main__":
    asyncio.run(generate_audio())
