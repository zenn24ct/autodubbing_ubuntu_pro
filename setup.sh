#!/bin/bash
set -e
echo "=== 日本語→英語 自動吹き替えシステム Pro セットアップ (Ubuntu) ==="

if ! command -v ffmpeg &>/dev/null; then
  echo "ffmpeg をインストール中..."
  sudo apt update -y && sudo apt install -y ffmpeg
fi
echo "✓ ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

if ! command -v python3 &>/dev/null; then
  sudo apt install -y python3 python3-pip python3-venv
fi
echo "✓ python3: $(python3 --version)"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "⚠️  .env を作成しました。ANTHROPIC_API_KEY を設定してください（日本語清書に使用）"
fi

echo ""
echo "✅ セットアップ完了"
echo "起動: bash run.sh"
