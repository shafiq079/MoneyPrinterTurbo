"""Import-safe subprocess entry point for one semantic scene-query attempt."""

import socket
import sys


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].isdecimal():
        return 2
    descriptor = int(sys.argv[1])
    if descriptor < 3:
        return 2
    from app.services import llm

    worker_socket = socket.socket(fileno=descriptor)
    llm.run_scene_query_worker(worker_socket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
