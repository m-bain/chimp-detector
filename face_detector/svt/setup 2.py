import setuptools

setuptools.setup(
    name="svt",
    version="2.0.1",
    author="Abhishek Dutta and Li Zhang",
    author_email="adutta@robots.ox.ac.uk, lz@robots.ox.ac.uk",
    description="Seebibyte Visual Tracker can be used to track any object in a video.",
    license="BSD",
    keywords='visual tracker, object tracker, svt, video tracker',
    long_description='SVT is a visual tracking software that can track any object in a video.',
    long_description_content_type="text/markdown",
    url="http://www.robots.ox.ac.uk/~vgg/projects/seebibyte/software/svt/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[], #['opencv-python', 'pytorch', 'torchvision'],
    entry_points={
        'console_scripts': [
            'svt=svt.main:main',
        ],
    },
#    data_files=[('svt/ui', ['svt/ui/svt.html'])]
    package_data={'svt':['ui/svt.html']}
)

