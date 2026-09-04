from pathlib import Path

import mujoco
import numpy as np

from dg5f_grasp_control.hand_model import (
    HAND_JOINT_NAMES,
    TACTILE_Y_ORIGIN_OFFSET_M,
)


class MujocoGravityCompensator:
    def __init__(self, model_xml_path):
        self.model = self._load_model(model_xml_path)
        self.data = mujoco.MjData(self.model)
        self.qadr, self.dadr = self._get_joint_addr()
        self.G = np.zeros(self.model.nv, dtype=np.float64)

    @staticmethod
    def _load_model(model_xml_path):
        path = Path(model_xml_path)
        if path.suffix != ".urdf":
            return mujoco.MjModel.from_xml_path(str(path))

        mesh_dir = (
            path.parent.parent
            / "meshes"
            / path.stem.removesuffix("_w_mount")
        )
        assets = {
            mesh_path.name: mesh_path.read_bytes()
            for mesh_path in mesh_dir.glob("*.STL")
        }
        return mujoco.MjModel.from_xml_string(
            path.read_text(encoding="utf-8"),
            assets,
        )

    def _get_joint_addr(self):
        qadr, dadr = [], []

        for name in HAND_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint not found: {name}")

            qadr.append(self.model.jnt_qposadr[jid])
            dadr.append(self.model.jnt_dofadr[jid])

        return np.array(qadr, dtype=int), np.array(dadr, dtype=int)

    def compute(self, q, gravity=None):
        if gravity is not None:
            self.model.opt.gravity[:] = np.asarray(gravity, dtype=np.float64)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qpos[self.qadr] = q

        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_rne(self.model, self.data, 0, self.G)

        return self.G[self.dadr].copy()

    def tactile_link_pose(self, q, finger):
        """Return the URDF/MuJoCo pose of the last non-tip finger link."""
        self.data.qpos[:] = 0.0
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        name = f"link_{int(finger)}_4"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            return None
        return (
            self.data.xpos[body_id].copy(),
            self.data.xmat[body_id].reshape(3, 3).copy(),
        )

    def tactile_contact_geometry(self, q, finger, x_mm, y_mm):
        """Return tactile surface point and outward normal in model-world coordinates."""
        self.data.qpos[:] = 0.0
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        finger = int(finger)
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, f"link_{finger}_tip"
        )
        geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, f"link_{finger}_tip_geom"
        )
        if body_id < 0 or geom_id < 0:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"link_{finger}_4"
            )
            geom_ids = np.flatnonzero(self.model.geom_bodyid == body_id)
            if body_id < 0 or geom_ids.size < 2:
                return None
            geom_id = int(max(geom_ids, key=lambda gid: self.model.geom_pos[gid, 0]))
            mesh_id = int(self.model.geom_dataid[geom_id])
            if mesh_id < 0:
                return None
            rotation = self.data.xmat[body_id].reshape(3, 3)
            position = self.data.geom_xpos[geom_id] - rotation @ self.model.mesh_pos[mesh_id]
        else:
            position = self.data.xpos[body_id]
            rotation = self.data.xmat[body_id].reshape(3, 3)
        ray_origin = position + rotation @ np.array(
            [
                TACTILE_Y_ORIGIN_OFFSET_M + float(y_mm) * 0.001,
                0.1,
                float(x_mm) * 0.001,
            ]
        )
        ray_direction = rotation @ np.array([0.0, -1.0, 0.0])
        normal = np.zeros(3, dtype=np.float64)
        distance = mujoco.mj_rayMesh(
            self.model,
            self.data,
            geom_id,
            ray_origin,
            ray_direction,
            normal,
        )
        if distance < 0.0:
            return None
        if np.dot(normal, -ray_direction) < 0.0:
            normal *= -1.0
        return ray_origin + distance * ray_direction, normal
