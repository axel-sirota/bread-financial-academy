#!/usr/bin/env python3
"""
Generate call recording audio files using OpenAI TTS for Week 6 lab.

Creates WAV audio files simulating customer service calls.
Uses different voices for agent and customer.

Usage:
    export OPENAI_API_KEY=your_key_here
    pip install openai pydub
    python generate_audio.py

Output:
    call_recordings/call_001.wav through call_005.wav
"""

import os
import json
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package required. Install with: pip install openai")
    exit(1)

try:
    from pydub import AudioSegment
except ImportError:
    print("ERROR: pydub package required. Install with: pip install pydub")
    print("Note: pydub also requires ffmpeg to be installed on your system")
    exit(1)

# Load transcripts
SCRIPT_DIR = Path(__file__).parent


def load_transcripts():
    """Load call transcripts from JSON file."""
    transcript_file = SCRIPT_DIR / "call_transcripts.json"
    if not transcript_file.exists():
        print("Transcripts file not found. Generating...")
        os.system(f"python {SCRIPT_DIR / 'generate_transcripts.py'}")

    with open(transcript_file) as f:
        return json.load(f)


# Call dialogue scripts (same as generate_transcripts.py)
CALL_DIALOGUES = {
    "call_001": [
        ("agent", "Thank you for calling First National Bank, this is Sarah speaking. How can I help you today?"),
        ("customer", "Yes, I'm calling about a charge on my account that I didn't make. I need to dispute it immediately."),
        ("agent", "I understand. I'm sorry to hear that. Can you please provide your account number so I can look into this?"),
        ("customer", "It's 4532-7891-2345-6789. The charge is for eight hundred forty seven dollars and ninety nine cents."),
        ("agent", "Thank you. I see the charge from Global Electronics for $847.99 on November 12th. Can you tell me more about why you're disputing this?"),
        ("customer", "I never made that purchase. I've never even heard of Global Electronics. I think someone stole my card information. This is really frustrating."),
        ("agent", "I completely understand your frustration, and I'm very sorry this happened to you. I'm going to flag this as potential fraud and start an investigation right away."),
        ("customer", "Thank you. What happens next? Will I get my money back? I'm really worried about this."),
        ("agent", "Yes, absolutely. I've placed a temporary hold on your card for security and initiated the dispute process. You should receive a provisional credit within 24 hours."),
        ("customer", "Okay, thank you for your help Sarah."),
        ("agent", "You're welcome. Thank you for calling First National Bank."),
    ],
    "call_002": [
        ("agent", "Good afternoon, First National Bank, this is Michael. How may I assist you?"),
        ("customer", "Hi, I just want to check my account balance please."),
        ("agent", "Certainly. May I have your account number?"),
        ("customer", "Sure, it's 9876-5432-1098-7654."),
        ("agent", "Thank you. Your checking account has a current balance of two thousand three hundred forty five dollars."),
        ("customer", "Okay, that sounds right. Thank you."),
        ("agent", "You're welcome. Is there anything else I can help you with?"),
        ("customer", "No, that's it. Thanks."),
        ("agent", "Thank you for calling First National Bank. Have a great day."),
    ],
    "call_003": [
        ("agent", "First National Bank, this is Jennifer. How can I help you?"),
        ("customer", "I lost my debit card! I think it might have been stolen. I need to cancel it right now!"),
        ("agent", "I understand this is stressful. Let me help you right away. Can you provide your account number?"),
        ("customer", "Yes, it's 1234-5678-9012-3456. Please hurry!"),
        ("agent", "I'm blocking the card right now. The card is now deactivated. No one can use it."),
        ("customer", "Thank goodness. Have there been any charges I didn't make?"),
        ("agent", "I see two suspicious charges. One at a gas station for forty five dollars and one at an electronics store for three hundred twenty dollars."),
        ("customer", "Those aren't mine! I haven't used my card today at all."),
        ("agent", "I'm marking both as fraudulent and starting a dispute. You'll receive a full refund within 24 hours."),
        ("customer", "Thank you so much for helping me quickly."),
    ],
    "call_004": [
        ("agent", "Thank you for calling First National Bank, this is David. How can I help you today?"),
        ("customer", "Hi David! I'm calling to say thank you. I had an issue with a duplicate charge last week and Maria helped me sort it out."),
        ("agent", "That's wonderful to hear! I'm glad we could resolve that for you."),
        ("customer", "Yes, the refund came through and everything looks correct now. I just wanted to share how happy I am with the service."),
        ("agent", "Thank you so much for taking the time to share that feedback. I'll make sure Maria gets recognized."),
        ("customer", "Please do! She was so helpful. Have a great day!"),
        ("agent", "You too! Thank you for being a valued customer."),
    ],
    "call_005": [
        ("agent", "First National Bank, this is Ana speaking. How may I help you?"),
        ("customer", "Hello, I need to make a wire transfer to Mexico. Can you help me?"),
        ("agent", "Of course! I'd be happy to help with an international wire transfer. May I have your account number?"),
        ("customer", "Yes, my account is 5555-4444-3333-2222."),
        ("agent", "To send a wire to Mexico, I'll need the recipient's name, bank name, and CLABE number."),
        ("customer", "The name is Carlos Rodriguez, the bank is Banco Nacional, and I want to send one thousand five hundred dollars."),
        ("agent", "Perfect. The wire transfer fee is thirty five dollars. The transfer will arrive within 1 to 2 business days."),
        ("customer", "That sounds good. Please proceed. Thank you for your help."),
        ("agent", "You're welcome. Thank you for choosing First National Bank."),
    ],
}

# Voice mapping - different voices for agent vs customer
VOICE_MAP = {
    "agent": "nova",      # Professional female voice for agents
    "customer": "onyx",   # Different voice for customers
}


def generate_audio_segment(client: OpenAI, text: str, voice: str) -> AudioSegment:
    """Generate audio for a single line of dialogue."""
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="mp3"
    )

    # Save to temp file and load as AudioSegment
    temp_file = "/tmp/temp_audio.mp3"
    response.stream_to_file(temp_file)

    audio = AudioSegment.from_mp3(temp_file)
    os.remove(temp_file)

    return audio


def generate_call_audio(client: OpenAI, call_id: str, dialogue: list, output_path: str):
    """Generate a complete call audio file from dialogue."""
    print(f"\n  Generating {call_id}...")

    combined = AudioSegment.silent(duration=500)  # Start with short silence

    for i, (speaker, text) in enumerate(dialogue):
        voice = VOICE_MAP[speaker]
        print(f"    [{i+1}/{len(dialogue)}] {speaker}: {text[:50]}...")

        try:
            segment = generate_audio_segment(client, text, voice)
            combined += segment
            combined += AudioSegment.silent(duration=800)  # Pause between speakers

            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"    ERROR generating segment: {e}")
            continue

    # Add ending silence
    combined += AudioSegment.silent(duration=500)

    # Export as WAV (Transcribe prefers WAV)
    combined.export(output_path, format="wav")

    duration = len(combined) / 1000
    print(f"    ✓ Generated: {output_path} ({duration:.1f}s)")

    return duration


def main():
    """Generate all call recording audio files."""
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Try loading from common locations
        env_files = [
            os.path.expanduser("~/repos/ollama-service/.env"),
            ".env",
            "../.env"
        ]
        for env_file in env_files:
            if os.path.exists(env_file):
                with open(env_file) as f:
                    for line in f:
                        if line.startswith("OPENAI_API_KEY="):
                            api_key = line.strip().split("=", 1)[1]
                            break
            if api_key:
                break

    if not api_key:
        print("ERROR: OPENAI_API_KEY not found")
        print("Set it with: export OPENAI_API_KEY=your_key")
        exit(1)

    # Initialize client
    client = OpenAI(api_key=api_key)

    # Create output directory
    output_dir = SCRIPT_DIR / "call_recordings"
    output_dir.mkdir(exist_ok=True)

    print(f"Generating {len(CALL_DIALOGUES)} call recordings using OpenAI TTS...")
    print(f"Output directory: {output_dir}")

    total_duration = 0

    for call_id, dialogue in CALL_DIALOGUES.items():
        output_path = output_dir / f"{call_id}.wav"
        duration = generate_call_audio(client, call_id, dialogue, str(output_path))
        total_duration += duration

    print(f"\n{'='*50}")
    print(f"Generated {len(CALL_DIALOGUES)} audio files")
    print(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"\nFiles:")
    for f in sorted(output_dir.glob("*.wav")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
