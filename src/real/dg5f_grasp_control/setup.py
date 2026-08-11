from glob import glob

from setuptools import find_packages, setup

package_name = "dg5f_grasp_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", [
            "config/grasp_real_common.yaml",
            "config/grasp_real_left_gains.yaml",
            "config/grasp_real_right_gains.yaml",
        ]),
        ("share/" + package_name + "/launch", [
            "launch/grasp_real.launch.py",
            "launch/grasp_right_compensation.launch.py",
            "launch/grasp_with_effort.launch.py",
            "launch/grasp_with_effort_right.launch.py",
        ]),
        ("share/" + package_name + "/models", ["models/dg5fs_left_w_mount.xml"]),
        ("share/" + package_name + "/models/meshes", glob("models/meshes/*.STL")),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="DG5F-S User",
    maintainer_email="user@example.com",
    description="DG5F-S grasp control package.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "grasp_real = dg5f_grasp_control.grasp_real_node:main",
        ],
    },
)
