#!/usr/bin/env python3

import os


if __name__ == "__main__":
    print("=== ENV READ START ===")
    print(f"FASTVIDEO_VSA_256={os.environ.get('FASTVIDEO_VSA_256', '<unset>')}")
    print(
        "FASTVIDEO_VSA_256_BACKEND="
        f"{os.environ.get('FASTVIDEO_VSA_256_BACKEND', '<unset>')}"
    )
    print(
        "FASTVIDEO_KERNEL_VSA_FORCE_TRITON="
        f"{os.environ.get('FASTVIDEO_KERNEL_VSA_FORCE_TRITON', '<unset>')}"
    )
    print(f"MY_FLAG={os.environ.get('MY_FLAG', '<unset>')}")
    print(f"MODAL_VSA256_RUN_CMD={os.environ.get('MODAL_VSA256_RUN_CMD', '<unset>')}")
    print(f"PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}")
    print("=== ENV READ END ===")
