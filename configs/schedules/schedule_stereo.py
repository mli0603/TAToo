# optimizer
max_iter = 25000
lr = 1e-6
optimizer = dict(
    type="AdamW",
    lr=lr,
    weight_decay=0.01
)
optimizer_config = dict(grad_clip=dict(max_norm=1))
lr_config = dict(
    policy="Step",
    by_epoch=False,
    gamma=0.1,
    step=[5000*3, 5000*4]
)
# runtime settings
runner = dict(type="IterBasedRunner", max_iters=max_iter)
checkpoint_config = dict(by_epoch=False, interval=2500)
evaluation = dict(interval=2500, metric="stereo_only")
