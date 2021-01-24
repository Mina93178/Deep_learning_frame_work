from setuptools import setup, find_packages
setup(
    name='NN',
    version='0.0.1',
    description='DL FrameWork to train and test on mnist and cifar10',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='https://github.com/Mina93178/Deep_learning_frame_work',
    author='MinaZ',
    author_email='mwagdy64@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python '
    ],
    keywords='Deep_learning_frame_work',
    packages=find_packages(),
    install_requires=["python3", "Pillow8.0.1", "matplotlib3.3.3", "numpy1.19.4", "pandas1.2.0",
                      "pip20.3.3", "psutil5.8.0", "pyinstaller4.1", "scipy1.6.0", "setuptools51.1.0", "twine"]

)