# optimizer
max_iter = 6250
optimizer = dict(type="Adam", lr=2e-4, weight_decay=1e-5)
optimizer_config = dict(grad_clip=dict(max_norm=1))
# learning policy
lr_config = dict(
    policy="OneCycle",
    max_lr=2e-4,
    total_steps=max_iter,
    pct_start=0.001,
    anneal_strategy="linear"
)
# runtime settings
runner = dict(type="IterBasedRunner", max_iters=max_iter)
checkpoint_config = dict(by_epoch=False, interval=1250)
evaluation = dict(interval=1250, metric="segmentation_only")
