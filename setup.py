from setuptools import setup, find_packages

setup(
    name='lsv-recognition',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "opencv-python <= 4.9.0.80",
        "scikit-learn <= 1.4.0",
        "mediapipe <= 0.10.9",
        "tensorflow <= 2.15.0.post1",
    ],
    author='Carlos Bone',
    author_email='cdbon15@gmail.com',
    description='Recognize up to 10 different gestures of the Venezuelan Sign Language',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/TSX-Corvo/lsv-recognition',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
)
