import os
import re
import sys
import versioneer  # noqa: E402
from setuptools import setup, find_packages

CLASSIFIERS = """
Development Status :: 4 - Beta
Programming Language :: Python :: 3
Programming Language :: Python :: 3.11
Programming Language :: Python :: 3.12
Programming Language :: Python :: 3.13
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Topic :: Scientific/Engineering :: Atmospheric Science
Topic :: Scientific/Engineering :: Oceanography
Topic :: Scientific/Engineering :: Physics
"""

MINIMUM_VERSIONS = {
    "numpy": "2.0.1",
    "xesmf": "0.8.9",
}

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)


def parse_requirements(reqfile):
    requirements = []

    with open(os.path.join(here, reqfile), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            parsed = re.match(r"(\w+)(.*?)(;.*)?$", line)
            pkg, deps, extra = parsed.groups()
            if extra is None:
                extra = ""
            deps = deps.replace("==", "<=")
            if pkg in MINIMUM_VERSIONS:
                deps = f"{deps},>={MINIMUM_VERSIONS[pkg]}"
            line = "".join([pkg, deps, extra])
            requirements.append(line)

    return requirements


INSTALL_REQUIRES = parse_requirements("requirements.txt")
EXTRAS_REQUIRE = {
    "test": ["pytest", "pytest-cov", "pytest-forked"],
}


with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

cmdclass = versioneer.get_cmdclass()

setup(
    name="veros",
    license="MIT",
    author="Roman Nuterman (NBI Copenhagen)",
    author_email="nuterman@nbi.ku.dk",
    keywords="earth systems geophysics python numpy",
    description="Versatile Earth-system coupler for atmosphere, ocean, sea-ice, and land components",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[c for c in CLASSIFIERS.split("\n") if c],
    zip_safe=False,
)
