from setuptools import find_packages, setup
from glob import glob
import os 

package_name = 'mpc_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'models'), glob('models/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vietvo',
    maintainer_email='vietvo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
  entry_points={
        'console_scripts': [
            'scenario_figure8 = mpc_bot.scenario_figure8:main',
            'scenario_square = mpc_bot.scenario_square:main',
            
            # 
            'scenario_obstacle = mpc_bot.scenario_obstacle:main',
            'scenario_nav = mpc_bot.scenario_nav:main',
            'scenario_nav_logging = mpc_bot.scenario_nav_logging:main',
            'scenario_figure8_obstacle = mpc_bot.scenario_figure8_obstacle:main',
        ],
    },
)
