import os

import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

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

    # Display in the viewer.
    v = Viewer()
    v.run_animations = True
    v.scene.camera.position = np.array([2.8, 2.3, -4.5])
    v.scene.add(seq1, seq2)
    v.run()
