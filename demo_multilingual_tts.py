#!/usr/bin/env python3
"""
Unified Multilingual Bidirectional Translation & Text-to-Speech (TTS) Demo CLI
==============================================================================
Features:
1. Bidirectional Text Translation:
   - English -> Nyishi / Apatani (Forward)
   - Nyishi / Apatani -> English (Reverse / Vice-Versa)
2. Neural Text-to-Speech (TTS):
   - Synthesizes spoken Apatani audio (.wav) directly from text using fine-tuned SpeechT5 + HiFi-GAN.
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


def run_translation(
    text: str,
    direction: str = "en2nyishi",
    model_dir: str = "./best_bidirectional_model",
) -> str:
    """Translates text in either direction using the single model."""
    if not os.path.exists(model_dir):
        if os.path.exists("./best_model"):
            model_dir = "./best_model"
        else:
            raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if direction == "en2nyishi":
        src_lang = "eng_Latn"
        tgt_lang = "hin_Deva"
    else:  # nyishi2en
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


def run_tts(
    text: str,
    output_path: str = "demo_synthesized_apatani.wav",
    tts_model_dir: str = "./best_apatani_tts",
) -> str:
    """Synthesizes Apatani text into spoken audio (.wav)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(tts_model_dir):
        processor = SpeechT5Processor.from_pretrained(tts_model_dir)
        model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_dir).to(device)
        spk_path = os.path.join(tts_model_dir, "speaker_embedding.pt")
        speaker_embedding = torch.load(spk_path, map_location=device, weights_only=True) if os.path.exists(spk_path) else torch.randn(1, 512, device=device) * 0.05
    else:
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
        speaker_embedding = torch.randn(1, 512, device=device) * 0.05

    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    model.eval()
    vocoder.eval()

    inputs = processor(text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

    sf.write(output_path, speech.cpu().numpy(), samplerate=16000)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Unified Bidirectional Translation & TTS Demo CLI")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["translate_forward", "translate_reverse", "tts", "demo"],
        help="Operation mode: 'translate_forward' (En->Nyishi), 'translate_reverse' (Nyishi->En), 'tts' (Text-to-Speech), or 'demo' (All).",
    )
    parser.add_argument(
        "--text_en",
        type=str,
        default="Noah, Shem, Ham, and Japheth.",
        help="English input text for forward translation.",
    )
    parser.add_argument(
        "--text_ind",
        type=str,
        default="Noa, Sem, Ham, ho Japhet.",
        help="Indigenous (Nyishi/Apatani) text for reverse translation or TTS.",
    )
    parser.add_argument(
        "--tts_output",
        type=str,
        default="demo_synthesized_apatani.wav",
        help="Output wav filename for TTS synthesis.",
    )
    args = parser.parse_args()

    print("\n==================================================================")
    print("   Arunachal AI Suite: Bidirectional Translation & Text-to-Speech ")
    print("==================================================================")

    if args.mode in ["translate_forward", "demo"]:
        print("\n[1] Forward Translation (English -> Indigenous):")
        print(f"    Source (English):      {args.text_en}")
        try:
            trans = run_translation(args.text_en, direction="en2nyishi")
            print(f"    Translated Output:     {trans}")
        except Exception as e:
            print(f"    Translation note: {e}")

    if args.mode in ["translate_reverse", "demo"]:
        print("\n[2] Reverse Translation (Indigenous -> English - Vice Versa):")
        print(f"    Source (Indigenous):   {args.text_ind}")
        try:
            trans_rev = run_translation(args.text_ind, direction="nyishi2en")
            print(f"    Translated to English: {trans_rev}")
        except Exception as e:
            print(f"    Translation note: {e}")

    if args.mode in ["tts", "demo"]:
        print("\n[3] Neural Text-to-Speech (TTS):")
        print(f"    Input Apatani Text:    {args.text_ind}")
        try:
            wav_file = run_tts(args.text_ind, output_path=args.tts_output)
            print(f"    Synthesized Audio:     {wav_file} (16 kHz, Mono)")
        except Exception as e:
            print(f"    TTS note: {e}")

    print("\n==================================================================\n")


if __name__ == "__main__":
    main()
