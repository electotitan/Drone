from setuptools import setup

package_name = 'circle_drawer'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sahil',
    maintainer_email='you@example.com',
    description='Draws a circle with turtlesim and returns the turtle to the center',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'draw_circle = circle_drawer.draw_circle:main',
        ],
    },
)
