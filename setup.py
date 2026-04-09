from setuptools import setup, find_packages

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
)
