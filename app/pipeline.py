"""
処理パイプライン — 日本語→英語 自動吹き替え Pro

STEP 1: 音声抽出 + Whisper 日本語文字起こし
STEP 2: Claude API で日本語清書（フィラー除去・誤認識修正）
         → APIキーなしの場合はスキップ
STEP 3: ユーザーがWeb画面で日本語を確認・編集
STEP 4: Google翻訳（日本語→英語）
STEP 5: edge-tts で英語音声生成 + 話速調整（最大2倍速）
STEP 6: 動画に英語音声を合成 → output.mp4
STEP 7: SRT 字幕生成 → subtitle.srt
"""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path

import whisper
import edge_tts
from deep_translator import GoogleTranslator
from pydub import AudioSegment

JOBS_DIR = Path("jobs")

# ── 英語 TTS 声優 ─────────────────────────────────────────────────────
VOICES = {
    "female":  "en-US-JennyNeural",
    "male":    "en-US-GuyNeural",
    "female2": "en-GB-SoniaNeural",
    "male2":   "en-GB-RyanNeural",
}

# 速度調整の上限
MAX_SPEED = 2.0

# 文末句読点
_SENTENCE_FINAL = frozenset('。！？!?…')
_CONTINUATION_SUFFIXES = (
    'の', 'て', 'で', 'が', 'を', 'に', 'は', 'も', 'へ', 'と',
    'けど', 'けれど', 'し', 'から', 'ので', 'たら', 'なら',
    'という', 'として', 'において',
)


# ── ステータス管理 ────────────────────────────────────────────────────
def update_status(job_id: str, status: str, progress: int, message: str) -> None:
    path = JOBS_DIR / job_id / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"status": status, "progress": progress, "message": message},
            f, ensure_ascii=False,
        )
    print(f"[{job_id}] [{progress:3d}%] {message}")


# ── 音声抽出 ─────────────────────────────────────────────────────────
def extract_audio(input_path: str, audio_path: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"音声抽出エラー: {result.stderr[-500:]}")


# ── 動画の長さ取得 ────────────────────────────────────────────────────
def get_duration(file_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", file_path],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


# ── セグメント統合（文単位） ──────────────────────────────────────────
def merge_into_sentences(segments: list[dict], max_sec: float = 15.0) -> list[dict]:
    if not segments:
        return segments

    merged: list[dict] = []
    buf_text  = ""
    buf_start = segments[0]["start"]
    buf_end   = segments[0]["end"]

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if not buf_text:
            buf_start = seg["start"]

        buf_text += text
        buf_end   = seg["end"]
        duration  = buf_end - buf_start

        has_sentence_end = buf_text[-1] in _SENTENCE_FINAL
        too_long         = duration >= max_sec
        is_continuation  = any(buf_text.rstrip('、,').endswith(s)
                               for s in _CONTINUATION_SUFFIXES)

        if (has_sentence_end or too_long) and not (is_continuation and not too_long):
            merged.append({
                "start": round(buf_start, 2),
                "end":   round(buf_end,   2),
                "text":  buf_text.strip(),
            })
            buf_text = ""

    if buf_text.strip():
        merged.append({
            "start": round(buf_start, 2),
            "end":   round(buf_end,   2),
            "text":  buf_text.strip(),
        })
    return merged


# ── STEP 2: Claude API で日本語清書 ──────────────────────────────────
def polish_segments_claude(segments: list[dict], api_key: str) -> list[dict]:
    """
    Whisperの生出力を英語翻訳に最適な日本語に清書する。
    ・話し言葉フィラーの除去（「ですね」「えー」「なんか」等）
    ・誤認識の修正ヒント（文脈から明らかなもの）
    ・省略された主語を補う（英語翻訳で主語が必要なため）
    ・自然な書き言葉に統一
    ・markdownコードブロックを確実に除去してパース
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    polished = []
    batch_size = 20

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start:batch_start + batch_size]

        items = [
            {"id": i + batch_start, "ja": seg["text"]}
            for i, seg in enumerate(batch)
        ]

        prompt = (
            "以下は日本語動画のWhisper文字起こしセグメントのリスト（JSON配列）です。\n"
            "このテキストは英語に翻訳し、動画の英語吹き替え音声として使用します。\n\n"
            "以下のルールで日本語を清書してください：\n"
            "1. 「ですね」「えー」「あの」「なんか」「まあ」などのフィラーを除去\n"
            "2. 省略された主語を補う（英語では主語が必須なため）\n"
            "3. 話し言葉を自然な書き言葉に変換\n"
            "4. 繰り返し表現をまとめて簡潔にする\n"
            "5. 英語に訳しやすい平易な表現に整える\n"
            "6. 意味・内容は変えない（清書のみ）\n\n"
            "出力は同じ構造のJSON配列で、jaフィールドのみ変更してください。\n"
            "説明不要、JSONのみ出力してください。\n\n"
            f"{json.dumps(items, ensure_ascii=False)}"
        )

        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            # markdownコードブロックを除去してからパース
            raw = response.content[0].text.strip()
            print(f"[Claude清書 応答先頭] {raw[:100]}")
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw = "\n".join(lines).strip()

            result_items = json.loads(raw)
            refined_map = {item["id"]: item["ja"] for item in result_items}

            for i, seg in enumerate(batch):
                seg_id = i + batch_start
                refined_text = refined_map.get(seg_id, seg["text"])
                print(f"[清書済み seg{seg_id}] {seg['text'][:30]} → {refined_text[:30]}")
                polished.append({**seg, "text": refined_text})

        except Exception as e:
            print(f"[Claude清書エラー (batch {batch_start})] {e}")
            import traceback; traceback.print_exc()
            polished.extend(batch)

    return polished


# ── STEP 1: Whisper 日本語文字起こし ─────────────────────────────────
def run_transcription(job_id: str, input_path: str, model_size: str = "large-v3") -> None:
    try:
        update_status(job_id, "transcribing", 5, "音声を抽出中...")

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_wav = os.path.join(tmpdir, "audio.wav")
            extract_audio(input_path, audio_wav)

            model_size = os.environ.get("WHISPER_MODEL", model_size)
            update_status(job_id, "transcribing", 15,
                          f"Whisper ({model_size}) で日本語を文字起こし中...（数分かかります）")

            model = whisper.load_model(model_size)
            result = model.transcribe(
                audio_wav,
                language="ja",
                task="transcribe",
                verbose=False,
                initial_prompt="以下は日本語の講義・会話です。文末には句点（。）を付けて書き起こしてください。",
            )

        raw_segments = [
            {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
            for s in result["segments"] if s["text"].strip()
        ]
        segments = merge_into_sentences(raw_segments)

        # STEP2: Claude清書（APIキーがあれば）
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if api_key:
            update_status(job_id, "polishing", 75,
                          "Claude API で日本語を清書中...（フィラー除去・文章整形）")
            segments = polish_segments_claude(segments, api_key)
            update_status(job_id, "polishing", 90, "清書完了")
        else:
            update_status(job_id, "transcribing", 85,
                          "ANTHROPIC_API_KEY 未設定のため清書スキップ")

        out = JOBS_DIR / job_id / "segments_ja.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        update_status(job_id, "ready_to_edit", 100,
                      f"文字起こし・清書完了（{len(segments)} セグメント）。日本語を確認・編集できます。")

    except Exception as e:
        update_status(job_id, "error", 0, f"文字起こしエラー: {e}")
        raise


# ── STEP 4: Google翻訳（日本語→英語） ───────────────────────────────
def translate_text(text: str) -> str:
    try:
        return GoogleTranslator(source="ja", target="en").translate(text)
    except Exception as e:
        print(f"[翻訳エラー] {e}")
        return text


# ── STEP 5: edge-tts ─────────────────────────────────────────────────
async def _tts_async(text: str, output_path: str, voice: str) -> None:
    await edge_tts.Communicate(text, voice).save(output_path)


def tts_segment_sync(text: str, output_path: str, voice: str) -> None:
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_tts_async(text, output_path, voice))
    finally:
        loop.close()


# ── 話速調整（最大2倍速） ────────────────────────────────────────────
def _build_atempo(speed: float) -> str:
    parts: list[str] = []
    r = speed
    while r > 2.0:
        parts.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        parts.append("atempo=0.5")
        r *= 2.0
    parts.append(f"atempo={r:.4f}")
    return ",".join(parts)


def adjust_speed(audio: AudioSegment, target_ms: float) -> AudioSegment:
    current_ms = len(audio)
    if current_ms == 0 or target_ms <= 0:
        return audio

    speed = current_ms / target_ms
    if speed > MAX_SPEED:
        print(f"[速度調整] {speed:.2f}x > 上限{MAX_SPEED}x → そのまま流す")
        return audio
    if speed < 1.05:
        return audio

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_in = f.name
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_out = f.name
    try:
        audio.export(tmp_in, format="mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in, "-filter:a", _build_atempo(speed), tmp_out],
            check=True, capture_output=True,
        )
        return AudioSegment.from_mp3(tmp_out)
    finally:
        os.unlink(tmp_in)
        os.unlink(tmp_out)


# ── SRT 字幕生成 ─────────────────────────────────────────────────────
def _sec_to_srt(s: float) -> str:
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    ss = int(s % 60)
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{ss:02d},{ms:03d}"


def _wrap_subtitle(text: str, lang: str = "en") -> str:
    """長い字幕を最大2行に分割する。"""
    # 言語ごとの1行あたり最大文字数
    max_chars = 20 if lang == "ja" else 42
    text = text.strip()
    if len(text) <= max_chars:
        return text

    # スペース（英語）または句読点（日本語）で折り返し
    if lang == "ja":
        # 日本語: max_chars文字ごとに分割
        lines = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    else:
        # 英語: 単語境界で折り返し
        import textwrap
        lines = textwrap.wrap(text, width=max_chars)

    # 最大2行まで
    return "\n".join(lines[:2])


def generate_srt(segments: list[dict], output_path: str, lang: str = "en") -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            wrapped = _wrap_subtitle(text, lang)
            # SRT規格: 番号 → タイムコード → テキスト → 空行
            f.write(f"{idx}\n")
            f.write(f"{_sec_to_srt(seg['start'])} --> {_sec_to_srt(seg['end'])}\n")
            f.write(f"{wrapped}\n")
            f.write("\n")  # エントリ間の空行（必須）
            idx += 1


# ── フルパイプライン（翻訳→TTS→動画合成） ───────────────────────────
def run_pipeline(job_id: str, voice_key: str = "female", make_subtitle: bool = True, subtitle_lang: str = "en") -> None:
    try:
        job_dir = JOBS_DIR / job_id
        voice   = VOICES.get(voice_key, VOICES["female"])

        # 編集済みを優先
        edited   = job_dir / "segments_ja_edited.json"
        original = job_dir / "segments_ja.json"
        seg_path = edited if edited.exists() else original

        if not seg_path.exists():
            raise RuntimeError("日本語セグメントファイルが見つかりません")

        with open(seg_path, encoding="utf-8") as f:
            ja_segments = json.load(f)

        input_files = [
            p for p in job_dir.iterdir()
            if p.stem == "original"
            and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".mp3", ".wav", ".m4a"}
        ]
        if not input_files:
            raise RuntimeError("元ファイルが見つかりません")

        input_path    = str(input_files[0])
        is_audio_only = input_files[0].suffix.lower() in {".mp3", ".wav", ".m4a"}
        total_duration = get_duration(input_path)
        total          = len(ja_segments)

        # 字幕生成（英語翻訳テキストで）
        if make_subtitle:
            update_status(job_id, "processing", 2, "字幕ファイルを生成中...")

        # STEP4: 翻訳
        update_status(job_id, "processing", 5, "日本語→英語 翻訳中...")
        en_segments = []
        for i, seg in enumerate(ja_segments):
            en_text = translate_text(seg["text"].strip()) if seg["text"].strip() else ""
            en_segments.append({
                "start": seg["start"],
                "end":   seg["end"],
                "text":  en_text,
            })
            update_status(job_id, "processing",
                          int(5 + (i + 1) / total * 20),
                          f"翻訳中 ({i + 1}/{total})")

        with open(job_dir / "segments_en.json", "w", encoding="utf-8") as f:
            json.dump(en_segments, f, ensure_ascii=False, indent=2)

        if make_subtitle:
            # 字幕言語に応じてセグメントを選択
            if subtitle_lang == "en":
                sub_segments = en_segments
                sub_lang = "en"
            else:
                # ja: 編集済み優先 → 清書済み → 生出力
                edited_path = job_dir / "segments_ja_edited.json"
                ja_path     = job_dir / "segments_ja.json"
                sub_path    = edited_path if edited_path.exists() else ja_path
                with open(sub_path, encoding="utf-8") as f:
                    sub_segments = json.load(f)
                sub_lang = "ja"
            generate_srt(sub_segments, str(job_dir / "subtitle.srt"), lang=sub_lang)

        # STEP5: TTS
        update_status(job_id, "processing", 30,
                      f"英語音声を生成中（声: {voice_key}）...")

        with tempfile.TemporaryDirectory() as tmpdir:
            track = AudioSegment.silent(duration=int(total_duration * 1000) + 3000)

            for i, seg in enumerate(en_segments):
                text = seg.get("text", "").strip()
                if not text:
                    continue

                start_ms = int(seg["start"] * 1000)
                end_ms   = int(seg["end"]   * 1000)
                seg_dur  = end_ms - start_ms

                tts_path = os.path.join(tmpdir, f"seg_{i:04d}.mp3")
                try:
                    tts_segment_sync(text, tts_path, voice)
                except Exception as e:
                    print(f"[TTS失敗 seg {i}] {e}")
                    continue

                tts_audio = AudioSegment.from_mp3(tts_path)

                if seg_dur > 0 and len(tts_audio) > seg_dur * 1.05:
                    tts_audio = adjust_speed(tts_audio, seg_dur)

                track = track.overlay(tts_audio, position=start_ms)

                update_status(
                    job_id, "processing",
                    int(30 + (i + 1) / total * 55),
                    f"音声生成中 ({i + 1}/{total}): {text[:30]}...",
                )

            # 音声書き出し
            update_status(job_id, "processing", 88, "英語音声トラックを書き出し中...")
            en_wav = os.path.join(tmpdir, "english_track.wav")
            track.export(en_wav, format="wav")

            if is_audio_only:
                import shutil
                shutil.copy(en_wav, str(job_dir / "output.mp4"))
                update_status(job_id, "done", 100, "完成しました！")
                return

            # STEP6: 動画合成（音声差し替え）
            update_status(job_id, "processing", 92, "動画に英語音声を合成中...")
            merged_path = os.path.join(tmpdir, "merged.mp4")
            result = subprocess.run(
                ["ffmpeg", "-y",
                 "-i", input_path, "-i", en_wav,
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                 "-map", "0:v:0", "-map", "1:a:0",
                 "-shortest", merged_path],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"動画合成エラー: {result.stderr[-500:]}")

            # STEP7: 字幕焼き込み
            update_status(job_id, "processing", 96, "字幕を動画に焼き込み中...")
            srt_path = str(job_dir / "subtitle.srt")
            output_path = str(job_dir / "output.mp4")

            # SRTパスのコロンをエスケープ（ffmpegのsubtitlesフィルタ要件）
            srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")

            subtitle_result = subprocess.run(
                ["ffmpeg", "-y",
                 "-i", merged_path,
                 "-vf", f"subtitles='{srt_escaped}':force_style='FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1,Alignment=2,MarginV=40,WrapStyle=2'",
                 "-c:a", "copy",
                 output_path],
                capture_output=True, text=True,
            )
            if subtitle_result.returncode != 0:
                # 字幕焼き込み失敗時は字幕なしで出力
                print(f"[字幕焼き込み失敗] {subtitle_result.stderr[-300:]}")
                print("字幕なしで出力します...")
                import shutil
                shutil.copy(merged_path, output_path)

        update_status(job_id, "done", 100, "完成しました！字幕付き動画をダウンロードできます。")

    except Exception as e:
        update_status(job_id, "error", 0, f"エラー: {e}")
        raise
