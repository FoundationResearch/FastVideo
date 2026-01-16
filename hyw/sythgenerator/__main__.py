# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    """
    Convenience entrypoint:
      - `python -m hyw.sythgenerator` defaults to the legacy 2D circle generator
      - `python -m hyw.sythgenerator sythball ...` runs the 3D plane+sphere generator
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) > 0 and argv[0] in {"ball", "sythball"}:
        from hyw.sythgenerator.generate_ball_dataset import main as ball_main

        return ball_main(argv[1:])

    from hyw.sythgenerator.generate_circle_dataset import main as circle_main

    return circle_main(argv)


if __name__ == "__main__":
    main()


