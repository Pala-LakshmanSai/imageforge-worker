from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run(
        "imageforge_worker.app:create_app",
        factory=True,
        host="0.0.0.0",  # noqa: S104 - the RunPod HTTPS proxy is the network boundary.
        port=8000,
        workers=1,
        access_log=True,
    )


if __name__ == "__main__":
    main()
