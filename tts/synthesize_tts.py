#!/usr/bin/env python3
"""
Apatani Text-to-Speech (TTS) Synthesis CLI
==========================================
Converts any input Apatani text into spoken audio (.wav) using the fine-tuned
SpeechT5 model and HiFi-GAN neural vocoder.
"""

import argparse
import os
import sys
import soundfile as sf
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor


def synthesize_speech(
    text: str,
    output_path: str = "output_speech.wav",
    model_dir: str = "./best_apatani_tts",
    vocoder_name: str = "microsoft/speecht5_hifigan",
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(model_dir) and os.path.exists(os.path.join("..", model_dir)):
        model_dir = os.path.join("..", model_dir)

    if not os.path.exists(model_dir):
        # Fallback to base model if fine-tuning checkpoint not present
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
        speaker_embedding = torch.randn(1, 512, device=device) * 0.05
    else:
        processor = SpeechT5Processor.from_pretrained(model_dir)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_dir).to(device)
        spk_path = os.path.join(model_dir, "speaker_embedding.pt")
        if os.path.exists(spk_path):
            speaker_embedding = torch.load(spk_path, map_location=device, weights_only=True)
        else:
            speaker_embedding = torch.randn(1, 512, device=device) * 0.05

    vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name).to(device)
    model.eval()
    vocoder.eval()

    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embedding,
            vocoder=vocoder,
        )

    sf.write(output_path, speech.cpu().numpy(), samplerate=16000)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Synthesize Apatani text to speech audio.")
    parser.add_argument(
        "--text",
        type=str,
        default="Hopa Ngo nunumi lukoso, nunuka sangomi hena siiyo.",
        help="Apatani text to synthesize.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthesized_apatani.wav",
        help="Path for generated output .wav file.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./best_apatani_tts",
        help="Directory of fine-tuned TTS model.",
    )
    args = parser.parse_args()

    print(f"Synthesizing text: '{args.text}'")
    out_file = synthesize_speech(args.text, output_path=args.output, model_dir=args.model_dir)
    print(f"Speech successfully generated: {out_file}")


if __name__ == "__main__":
    main()
