#!/bin/bash
if [[ -n "$BASH_SOURCE" ]]; then
    SCRIPT_FILE=$(readlink -f "$BASH_SOURCE")
else
    SCRIPT_FILE=$(readlink -f "$0")
fi

TCP_PATH=$(realpath "$(dirname "$SCRIPT_FILE")/../..")

if [[ -d "$TCP_PATH" ]]; then
    export PYTHONPATH=$PYTHONPATH:$TCP_PATH
    echo "✅ 找到TCP目录：$TCP_PATH，PYTHONPATH已更新"
else
    echo "❌ 错误：$TCP_PATH 目录不存在！"
    echo "⚠️ PYTHONPATH 未更新"
    exit 1
fi
