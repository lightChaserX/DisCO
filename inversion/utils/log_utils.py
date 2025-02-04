import numpy as np
from PIL import Image
import wandb
from configs import global_config
import torch
import matplotlib.pyplot as plt
from skvideo.io import FFmpegWriter

def log_image_from_w(w, c, G, name):
    img = get_image_from_w(w, c, G)
    pillow_image = Image.fromarray(img)
    wandb.log(
        {f"{name}": [
            wandb.Image(pillow_image, caption=f"current inversion {name}")]},
        step=global_config.training_step)


def log_images_from_w(ws, cs, G, names):
    for name, w, c in zip(names, ws, cs):
        # w = w.to(device)
        # c = c.to(device)
        log_image_from_w(w, c, G, name)


def plot_image_from_w(w, G):
    img = get_image_from_w(w, G)
    pillow_image = Image.fromarray(img)
    plt.imshow(pillow_image)
    plt.show()


def plot_image(img):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    pillow_image = Image.fromarray(img[0])
    plt.imshow(pillow_image)
    plt.show()


def save_image(name, method_type, results_dir, image, run_id):
    image.save(f'{results_dir}/{method_type}_{name}_{run_id}.jpg')


def save_w(w, c, G, name, method_type, results_dir):
    im = get_image_from_w(w, c, G)
    im = Image.fromarray(im, mode='RGB')
    save_image(name, method_type, results_dir, im)


def save_concat_image(base_dir, image_latents, new_inv_image_latent, new_G,
                      old_G,
                      file_name,
                      extra_image=None):
    images_to_save = []
    if extra_image is not None:
        images_to_save.append(extra_image)
    for latent in image_latents:
        images_to_save.append(get_image_from_w(latent, old_G))
    images_to_save.append(get_image_from_w(new_inv_image_latent, new_G))
    result_image = create_alongside_images(images_to_save)
    result_image.save(f'{base_dir}/{file_name}.jpg')


def save_single_image(base_dir, image_latent, G, file_name):
    image_to_save = get_image_from_w(image_latent, G)
    image_to_save = Image.fromarray(image_to_save, mode='RGB')
    image_to_save.save(f'{base_dir}/{file_name}.jpg')


def create_alongside_images(images):
    res = np.concatenate([np.array(image) for image in images], axis=1)
    return Image.fromarray(res, mode='RGB')


def get_image_from_w(w, c, G):
    if len(w.size()) <= 2:
        w = w.unsqueeze(0)
    with torch.no_grad():
        if w.shape[1]!= G.backbone.mapping.num_ws:
            w = w.repeat([1, G.backbone.mapping.num_ws, 1])
        img = G.synthesis(w, c, noise_mode='const')["image"]
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    return img[0]


def to_img(img, as_torch=False, as_numpy=False, as_PIL=False):
    img = img["image"][0].permute(1, 2, 0) / 2 + 0.5
    if as_torch:
        return img
    img = img.detach().cpu().numpy()
    if as_numpy:
        return img
    if as_PIL:
        return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def save_gif(frames, out_path="video.mp4", display=False):
    if display:
        from tqdm import tqdm
        tq = tqdm
    else:
        tq = lambda x: x
    writer = FFmpegWriter(out_path)
    for frame in tq(frames):
        img = np.asarray(frame).copy()
        img.setflags(write=1)
        writer.writeFrame(img)
        # if display:
        #     from IPython.display import display as show
        #     show(frame)
    writer.close()