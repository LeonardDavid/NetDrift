import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setuptools.setup(
    name='racetrack',
    version="0.0.1",
    author='Mikail Yayla, Leonard David Bereholschi',
    author_email='mikail.yayla@tu-dortmund.de, leonard.bereholschi@tu-dortmund.de',
    description='RTM Misalignment Fault Injection for BNNs',
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    ext_modules=[
        CUDAExtension('racetrack', [
            'racetrack.cpp',
            'racetrack_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
