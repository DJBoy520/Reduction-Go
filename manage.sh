#!/bin/bash
# manage.sh — FPLL 项目后台管理脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/run.pid"
LOG_DIR="$SCRIPT_DIR/logs"
STDOUT_LOG="$LOG_DIR/stdout.log"

mkdir -p "$LOG_DIR"

# ── 辅助函数 ──

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        # PID 文件存在但进程已死，清理
        rm -f "$PID_FILE"
    fi
    return 1
}

usage() {
    echo "用法: $0 {start|stop|status} [参数...]"
    echo ""
    echo "  start [params]  后台启动 python main.py [params]"
    echo "  stop            停止运行中的进程"
    echo "  status          查看运行状态"
    echo ""
    echo "示例:"
    echo "  $0 start"
    echo "  $0 start toy --verbose"
    echo "  $0 start --no-bkz --bkz-block-size 10"
}

# ── start ──

cmd_start() {
    if is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        echo "程序已在运行 (PID: $pid)，请先执行 $0 stop"
        return 1
    fi

    echo "启动 python main.py $* ..."
    nohup python3 main.py "$@" > "$STDOUT_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 0.5

    if kill -0 "$pid" 2>/dev/null; then
        echo "启动成功 (PID: $pid)"
        echo "日志: $LOG_DIR/attack.log"
        echo "标准输出: $STDOUT_LOG"
    else
        echo "启动失败，请检查 $STDOUT_LOG"
        rm -f "$PID_FILE"
        return 1
    fi
}

# ── stop ──

cmd_stop() {
    if ! is_running; then
        echo "程序未运行"
        return 0
    fi

    local pid
    pid=$(cat "$PID_FILE")
    echo "停止进程 $pid ..."

    kill "$pid" 2>/dev/null || true

    # 等待最多 3 秒
    local waited=0
    while [ $waited -lt 3 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "进程已终止"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    # 强制杀死
    echo "强制终止进程 $pid ..."
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "进程已强制终止"
}

# ── status ──

cmd_status() {
    if is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        local runtime
        runtime=$(ps -o etime= -p "$pid" 2>/dev/null | xargs)
        echo "程序运行中 (PID: $pid, 运行时间: ${runtime:-未知})"

        # 最近 5 行日志
        if [ -f "$LOG_DIR/attack.log" ]; then
            echo ""
            echo "最近日志:"
            tail -n 5 "$LOG_DIR/attack.log"
        fi
    else
        echo "程序未运行"

        # 上次运行信息
        if [ -f "$LOG_DIR/attack.log" ]; then
            local mtime
            mtime=$(stat -c %y "$LOG_DIR/attack.log" 2>/dev/null | cut -d. -f1)
            echo "上次运行时间: ${mtime:-未知}"
            echo ""
            echo "上次运行最后日志:"
            tail -n 5 "$LOG_DIR/attack.log"
        else
            echo "日志文件尚未生成"
        fi
    fi
}

# ── 主入口 ──

case "${1:-}" in
    start)
        shift
        cmd_start "$@"
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    *)
        usage
        exit 1
        ;;
esac
