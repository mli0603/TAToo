# model settings
max_disp = 256

model = dict(
    type='TAToo',
    stereo=dict(
        type='HITNetMF',
        backbone=dict(
            type='HITUNet',
        ),
        initialization=dict(
            type='TileInitialization',
            max_disp=max_disp,
        ),
        propagation=dict(
            type='TilePropagation',
        ),
        loss=dict(
            type='HITLoss',
            max_disp=max_disp,
            alpha=0.9,
            c=0.1,
            loss_weight=1.0
        ),
    ),
    # stereo=dict(
    #     type='CREStereoMF',
    #     crestereo=dict(
    #         type='CREStereo',
    #         max_disp=max_disp,
    #         pretained='PATH_TO/crestereo_eth3d.pth' # https://drive.google.com/file/d/1D2s1v4VhJlNz98FQpFxf_kBAKQVN_7xo/view
    #     ),
    #     loss=dict(
    #         type='FlowLoss',
    #         negate=False,
    #         gamma=0.8
    #     )
    # ),
    segmentation=dict(
        type='LinkNet34',
        num_classes=3,
        loss=dict(
            type="SegmentationLoss",
            loss_weight=1.0
       )
    ),
    motion=dict(
        type='Motion',
        config=dict(
            optimize_flow_weight=True,
            optimize_disp_weight=True,
            steps=6,
            gn_step=1,
            optimize_d_scale=500 * 0.06,
            delta_d_scale=1.0,
            sample_target_disp=True
        ),
        loss=dict(
            type='MotionLoss',
            geo_weight=10.0,
            res_weight=0.1,
            flo_weight=0.1,
            w_tau=1.0,
            w_phi=1.0,
            gamma=0.3
        ),
    ),
    test_cfg=dict(mode='whole')
)
