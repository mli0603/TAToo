# optimizer
max_iter = 50000
lr = 1e-6
optimizer = dict(
    type="AdamW",
    lr=lr,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'stereo': dict(lr_mult=0.05, decay_mult=0.05),
            'segmentation': dict(lr_mult=0.05, decay_mult=0.05)
        }
    )
)
optimizer_config = dict(grad_clip=dict(max_norm=1))
lr_config = dict(
    policy="Step",
    by_epoch=False,
    gamma=0.1,
    step=[40000, 45000]
)
# runtime settings
runner = dict(type="IterBasedRunner", max_iters=max_iter)
checkpoint_config = dict(by_epoch=False, interval=2500)
evaluation = dict(interval=1250, metric="motion_only")
