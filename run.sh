#!/bin/bash
set -e

if [ ! -d ".venv" ]; then
  echo "❌ 仮想環境が見つかりません。先に bash setup.sh を実行してください。"
  exit 1
fi

source .venv/bin/activate

if [ -f ".env" ]; then
  export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

PORT=${PORT:-8000}

echo "🚀 日本語→英語 自動吹き替えシステム Pro"
echo "   http://localhost:${PORT}"
echo "   停止: Ctrl+C"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
