import codecs
import os

from setuptools import find_packages, setup

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
]

INSTALL_REQUIRES = [
    "jax>=0.4.1",
    "jaxlib>=0.4.1",
    "simple-pytree==0.1.6",
]

EXTRA_REQUIRE = {
    "dev": ["pytest", "pre-commit", "pytest-cov", "flax"],
}

GLOBAL_PATH = os.path.dirname(os.path.realpath(__file__))


def read(*local_path: str) -> str:
    """Read a file, given a local path.

    Args:
        *local_path (str): The local path to the file.

    Returns:
        str: The contents of the file.
    """
    with codecs.open(os.path.join(GLOBAL_PATH, *local_path), "rb", "utf-8") as f:
        return f.read()


if __name__ == "__main__":
    setup(
        name="mytree",
        version="0.2.1",
        author="Daniel Dodd",
        author_email="daniel_dodd@icloud.com",
        license="MIT",
        description="Module pytrees that cleanly handle parameter trainability and transformations for JAX models.",
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        packages=find_packages(".", exclude=["tests"]),
        python_requires=">=3.8",
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRA_REQUIRE,
        zip_safe=True,
    )
