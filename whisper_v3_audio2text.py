from faster_whisper import WhisperModel

model_size = "large-v3"

# Initialize the WhisperModel
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Transcribe audio and get segments
segments, info = model.transcribe(
    "T_saginata.mp3",
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)

# Open a file to save the text
with open("T_saginata_text.txt", "w") as f:
    # Write detected language and probability to the file
    f.write("Detected language '%s' with probability %f\n\n" % (info.language, info.language_probability))
    
    # Write each segment to the file
    for segment in segments:
        #f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
        f.write("%s\n" % (segment.text))

print("Transcribed text saved to 'T_saginata.txt'")
