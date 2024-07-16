import cv2
import os
import numpy as np
import torch

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.headless import HeadlessRenderer

if __name__ == '__main__':
    # Load some data to simulate that we only have one frame available at a time.
    npz_data_path = os.path.join(C.datasets.amass, "CNRS/283/0_L_1_stageii.npz")

    body_data = np.load(npz_data_path)
    smpl_layer = SMPLLayer(model_type='smplh', gender=body_data['gender'].item(), device=C.device)

    poses_root = body_data['poses'][:, :3]
    poses_body = body_data['poses'][:, 3:3 + smpl_layer.bm.NUM_BODY_JOINTS * 3]
    betas = body_data['betas'][np.newaxis]
    trans = body_data['trans']

    # Create a SMPLSequence with just one frame.
    smpl_seq = SMPLSequence(smpl_layer=smpl_layer,
                            poses_root=poses_root[0:1],
                            poses_body=poses_body[0:1],
                            betas=betas,
                            trans=trans[0:1],
                            z_up=True)

    # Create the headless renderer and add the sequence.
    v = HeadlessRenderer()
    v.scene.add(smpl_seq)

    for f in range(poses_root.shape[0]):
        # Load the next available frame data and update the SMPL sequence.
        smpl_seq.poses_body = torch.from_numpy(poses_body[f:f + 1]).float().to(C.device)
        smpl_seq.poses_root = torch.from_numpy(poses_root[f:f + 1]).float().to(C.device)
        smpl_seq.trans = torch.from_numpy(trans[f:f + 1]).float().to(C.device)
        smpl_seq.redraw()

        # Get the frame and display it in an OpenCV window.
        img = v.get_frame()
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow('Frame', img)
        cv2.waitKey(333)