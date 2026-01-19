import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import run_nerf
from load_blender import load_blender_data
from run_nerf_helpers import get_rays, img2mse, mse2psnr


def _downsample_images(images, factor):
    if factor <= 1:
        return images, 1.0

    h, w = images.shape[1:3]
    new_h = max(1, h // factor)
    new_w = max(1, w // factor)
    images = tf.image.resize(
        images, [new_h, new_w], method=tf.image.ResizeMethod.AREA
    ).numpy()
    scale = new_w / float(w)
    return images, scale


def run_training_demo(
    config_path="./logs/lego_example/config.txt",
    max_steps=200,
    log_every=50,
    n_rand=1024,
    downscale=8,
    max_train_images=6,
    seed=0,
):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    parser = run_nerf.config_parser()
    args = parser.parse_args(["--config", config_path])

    # Keep the demo fast and avoid reloading pretrained weights.
    args.no_reload = True
    args.N_importance = 0
    args.N_samples = min(32, args.N_samples)
    args.N_rand = n_rand
    args.no_batching = True

    if args.dataset_type != "blender":
        raise ValueError("Training demo expects blender data from a config file.")

    images, poses, _render_poses, hwf, _i_split = load_blender_data(
        args.datadir, args.half_res, args.testskip
    )

    if args.white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]

    images, scale = _downsample_images(images, downscale)
    H, W, focal = hwf
    H = int(images.shape[1])
    W = int(images.shape[2])
    focal = float(focal) * scale

    train_count = min(max_train_images, images.shape[0] - 1)
    train_images = images[:train_count]
    train_poses = poses[:train_count]
    testimg = images[-1]
    testpose = poses[-1, :3, :4]

    render_kwargs_train, render_kwargs_test, _start, grad_vars, _models = run_nerf.create_nerf(
        args
    )
    render_kwargs_train.update({"near": tf.cast(2.0, tf.float32), "far": tf.cast(6.0, tf.float32)})
    render_kwargs_test.update({"near": tf.cast(2.0, tf.float32), "far": tf.cast(6.0, tf.float32)})

    optimizer = tf.keras.optimizers.Adam(args.lrate)

    psnrs = []
    iters = []
    start_t = time.time()

    coords = tf.stack(
        tf.meshgrid(tf.range(H), tf.range(W), indexing="ij"), axis=-1
    )
    coords = tf.reshape(coords, [-1, 2])
    pixel_count = H * W
    n_rand = min(n_rand, pixel_count)

    for step in range(max_steps + 1):
        img_i = np.random.randint(train_images.shape[0])
        target = train_images[img_i]
        pose = train_poses[img_i, :3, :4]

        rays_o, rays_d = get_rays(H, W, focal, pose)
        select_inds = np.random.choice(pixel_count, size=[n_rand], replace=False)
        select_coords = tf.gather(coords, select_inds)

        rays_o = tf.gather_nd(rays_o, select_coords)
        rays_d = tf.gather_nd(rays_d, select_coords)
        batch_rays = tf.stack([rays_o, rays_d], axis=0)
        target_s = tf.gather_nd(target, select_coords)

        with tf.GradientTape() as tape:
            rgb, disp, acc, extras = run_nerf.render(
                H,
                W,
                focal,
                chunk=args.chunk,
                rays=batch_rays,
                retraw=True,
                **render_kwargs_train,
            )
            loss = img2mse(rgb, target_s)
            if "rgb0" in extras:
                loss = loss + img2mse(extras["rgb0"], target_s)

        grads = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(grads, grad_vars))

        if step % log_every == 0:
            rgb, disp, acc, _ = run_nerf.render(
                H,
                W,
                focal,
                chunk=args.chunk,
                c2w=testpose,
                **render_kwargs_test,
            )
            psnr = mse2psnr(img2mse(rgb, testimg))
            psnrs.append(float(psnr.numpy()))
            iters.append(step)

            print("iter", step, "psnr", psnrs[-1])
            print("secs per iter", (time.time() - start_t) / max(1, log_every))
            start_t = time.time()

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(np.clip(rgb, 0.0, 1.0))
            plt.title(f"Iter {step}")
            plt.axis("off")

            plt.subplot(122)
            plt.plot(iters, psnrs)
            plt.title("PSNR")
            plt.tight_layout()
            plt.show()

    return {"iters": iters, "psnrs": psnrs}
