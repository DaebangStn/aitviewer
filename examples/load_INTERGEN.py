import os

import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.renderables.lines import Lines


if __name__ == "__main__":
    # Load an AMASS sequence and make sure it's sampled at 60 fps. This automatically loads the SMPL-H model.
    # We set transparency to 0.5 and render the joint coordinates systems.
    c = (149 / 255, 85 / 255, 149 / 255, 0.5)
    seq1, seq2 = SMPLSequence.from_intergen(
        pkl_data_path=os.path.join(C.datasets.amass, "inter1.pkl"),
        fps_out=60.0,
        color=c,
        name="AMASS Running",
        show_joint_angles=True,
    )

    line_shape = list(seq1.joints.shape)
    line_shape[1] *= 2
    line_pos = np.zeros(line_shape)
    line_pos[:, ::2, :] = seq1.joints
    line_pos[:, 1::2, :] = seq2.joints
    line_pos_z_up = line_pos[:, :, [0, 2, 1]]
    c = (0 / 255, 85 / 255, 200 / 255, 0.5)
    line_renderable = Lines(line_pos, color=c, r_base=0.004)

    # Display in the viewer.
    v = Viewer()
    v.run_animations = True
    v.scene.camera.position = np.array([1.5, 2, 2.5])
    v.scene.add(seq1, seq2)
    v.scene.add(line_renderable)
    v.run()
