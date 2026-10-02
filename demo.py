#!/usr/bin/env python3
"""
Interactive Demonstration CLI for Machine Translation and Text-to-Speech
Arunachal Pradesh Indigenous Language Technologies (Nyishi & Apatani)

Demonstrates:
1. Bidirectional Neural Machine Translation (English <-> Nyishi)
2. Neural Text-to-Speech Synthesis (Apatani)
"""

import argparse
import os
import sys
import soundfile as sf
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)


def translate(
    text: str,
    direction: str = "en2nyishi",
    model_dir: str = "./best_bidirectional_model",
) -> str:
    """Translates text between English and Nyishi using the bidirectional model."""
    if not os.path.exists(model_dir):
        if os.path.exists("./best_model"):
            model_dir = "./best_model"
        else:
            raise FileNotFoundError(f"Model checkpoint directory not found at: {model_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if direction == "en2nyishi":
        src_lang = "eng_Latn"
        tgt_lang = "hin_Deva"
    else:
        src_lang = "hin_Deva"
        tgt_lang = "eng_Latn"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang=src_lang, tgt_lang=tgt_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            num_beams=4,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def synthesize(
    text: str,
    output_path: str = "demo_synthesized.wav",
    tts_model_dir: str = "./best_apatani_tts",
) -> str:
    """Synthesizes Apatani speech waveform from input text."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = SpeechT5Processor.from_pretrained(tts_model_dir)
    model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_dir).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    model.eval()
    vocoder.eval()

    spk_path = os.path.join(tts_model_dir, "speaker_embedding.pt")
    if os.path.exists(spk_path):
        speaker_embedding = torch.load(spk_path, map_location=device, weights_only=True)
    else:
        speaker_embedding = torch.randn(1, 512, device=device) * 0.05

    inputs = processor(text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

    audio_np = speech.cpu().numpy()
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val * 0.95

    sf.write(output_path, audio_np, samplerate=16000)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Demonstration CLI for MT and TTS models.")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["translate_en2nyi", "translate_nyi2en", "tts", "all"],
        help="Target task to demonstrate.",
    )
    parser.add_argument(
        "--text_en",
        type=str,
        default="Noah, Shem, Ham, and Japheth.",
        help="Input English sentence for translation.",
    )
    parser.add_argument(
        "--text_nyi",
        type=str,
        default="Noa, Sem, Ham, ho Japhet.",
        help="Input Nyishi sentence for reverse translation.",
    )
    parser.add_argument(
        "--text_apa",
        type=str,
        default="Hopa Ngo nunumi lukoso, nunuka sangomi hena siiyo.",
        help="Input Apatani sentence for TTS synthesis.",
    )
    parser.add_argument(
        "--tts_output",
        type=str,
        default="synthesized_apatani.wav",
        help="Output audio path (.wav).",
    )
    args = parser.parse_args()

    print("------------------------------------------------------------")
    print("Hackathon Arunachal: Model Inference & Demonstration")
    print("------------------------------------------------------------")

    if args.task in ["translate_en2nyi", "all"]:
        print("[MT] English -> Nyishi Translation:")
        print(f"  Source (English): {args.text_en}")
        try:
            out_nyi = translate(args.text_en, direction="en2nyishi")
            print(f"  Target (Nyishi):  {out_nyi}")
        except Exception as e:
            print(f"  Error: {e}")

    if args.task in ["translate_nyi2en", "all"]:
        print("\n[MT] Nyishi -> English Translation (Reverse Direction):")
        print(f"  Source (Nyishi):  {args.text_nyi}")
        try:
            out_en = translate(args.text_nyi, direction="nyishi2en")
            print(f"  Target (English): {out_en}")
        except Exception as e:
            print(f"  Error: {e}")

    if args.task in ["tts", "all"]:
        print("\n[TTS] Apatani Neural Text-to-Speech Synthesis:")
        print(f"  Text:   {args.text_apa}")
        try:
            out_wav = synthesize(args.text_apa, output_path=args.tts_output)
            print(f"  Output: {out_wav} (16 kHz, Mono)")
        except Exception as e:
            print(f"  Error: {e}")

    print("------------------------------------------------------------")


if __name__ == "__main__":
    import numpy as np
    main()
