from setuptools import setup, find_packages
import glob

setup(
    name="turboquant",
    version="0.1.0",
    packages=find_packages(include=["turboquant*"]),
    python_requires=">=3.10",
    install_requires=["torch>=2.1.0"],
    entry_points={
        "vllm.general_plugins": [
            "turboquant = turboquant.vllm_plugin:register",
        ],
    },
    data_files=[
        ("csrc/src", glob.glob("csrc/src/*.cu")),
        ("csrc/include", glob.glob("csrc/include/*.cuh")),
        ("csrc/include/turboquant", glob.glob("csrc/include/turboquant/*.cuh")
         + glob.glob("csrc/include/turboquant/*.h")),
    ],
)
