from setuptools import find_packages, setup

package_name = 'roverlad_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lehongnhatminh',
    maintainer_email='lehongnhatminh@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'roverlad_cv = roverlad_control.roverlad_cv:main',
            'roverlad_imu = roverlad_control.roverlad_imu:main',
            'roverlad_run = roverlad_control.roverlad_run:main',
        ],
    },
)
