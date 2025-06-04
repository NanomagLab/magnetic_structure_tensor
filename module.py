import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from colorsys import hsv_to_rgb

def periodic_pad_2d(x, pad_i: int = 1, pad_j: int = 1):
    padded = tf.concat([x[..., -pad_i:, :], x, x[..., :pad_i, :] ], axis=-2)
    padded = tf.concat([padded[..., -pad_j:, :, :], padded, padded[..., :pad_j, :, :]], axis=-3)
    return padded

def zero_pad_2d(x, pad_i: int = 1, pad_j: int = 1):
    """
    :param x: 4-rank tensor
    :param pad_i:
    :param pad_j:
    :return: 4-rank tensor
    """
    padded = tf.pad(x, [[0, 0], [pad_i, pad_i], [pad_j, pad_j], [0, 0]])
    return padded

def pad_2d(x, pad_i: int = 1, pad_j: int = 1, padmode='periodic'):
    if padmode == 'periodic':
        return periodic_pad_2d(x, pad_i, pad_j)
    elif padmode == 'zeros':
        return zero_pad_2d(x, pad_i, pad_j)
    elif padmode == "valid":
        return x
    else:
        raise ValueError('padmode must be one of "periodic", "zeros", or "valid"')

def spin_to_texture_field(spins, padmode='periodic'):
    padded = pad_2d(spins, pad_i=1, pad_j=1, padmode=padmode)
    dsdx = (padded[..., 1:-1, 2:, :] - padded[..., 1:-1, :-2, :]) / 2.0
    dsdy = (-padded[..., 2:, 1:-1, :] + padded[..., :-2, 1:-1, :]) / 2.0
    chi_x = tf.linalg.cross(spins, dsdx)
    chi_y = tf.linalg.cross(spins, dsdy)
    return tf.stack([chi_x, chi_y], axis=-1)

def texture_field_to_antiskyrmion_components(texture_field):
    chi_alpha = (texture_field[..., 0, 0] - texture_field[..., 1, 1]) / 2.0
    chi_beta = (texture_field[..., 1, 0] + texture_field[..., 0, 1]) / 2.0
    return tf.stack([chi_alpha, chi_beta], axis=-1)

def texture_field_to_skyrmion_components(texture_field):
    chi_B = (texture_field[..., 0, 0] + texture_field[..., 1, 1]) / 2.0
    chi_N = (texture_field[..., 1, 0] - texture_field[..., 0, 1]) / 2.0
    return tf.stack([chi_B, chi_N], axis=-1)

def spin_to_antiskyrmion_components(spins, padmode='periodic'):
    texture_field = spin_to_texture_field(spins, padmode=padmode)
    antiskyrmion_components = texture_field_to_antiskyrmion_components(texture_field)
    return antiskyrmion_components

def antiskyrmion_components_to_defects(antiskyrmion_components, padmode='periodic'):
    angle = tf.atan2(antiskyrmion_components[..., 1], antiskyrmion_components[..., 0])
    padded = pad_2d(angle[..., tf.newaxis], pad_i=1, pad_j=1, padmode=padmode)[..., 0]
    defects = (padded[..., 1:-1, 2:] - padded[..., 1:-1, 1:-1] + np.pi) % (2. * np.pi) - np.pi
    defects += (padded[..., 2:, 2:] - padded[..., 1:-1, 2:] + np.pi) % (2. * np.pi) - np.pi
    defects += (padded[..., 2:, 1:-1] - padded[..., 2:, 2:] + np.pi) % (2. * np.pi) - np.pi
    defects += (padded[..., 1:-1, 1:-1] - padded[..., 2:, 1:-1]) % (2. * np.pi) - np.pi
    defects = (defects - 4. * np.pi) / (2. * np.pi) - np.pi
    return defects

def texture_field_to_defects(texture_field, padmode='periodic'):
    antiskyrmion_components = texture_field_to_antiskyrmion_components(texture_field)
    defects = antiskyrmion_components_to_defects(antiskyrmion_components, padmode=padmode)
    return defects

def spin_to_defects(spins, padmode='periodic'):
    texture_field = spin_to_texture_field(spins, padmode=padmode)
    defects = texture_field_to_defects(texture_field, padmode=padmode)
    return defects

def xy_to_rgb(xy):
    angle = np.arctan2(xy[..., 1], xy[..., 0])
    twilight = (angle % (2. * np.pi)) / (2. * np.pi)
    rgb = plt.get_cmap('twilight')(twilight)
    return rgb

def spin_to_rgb(spin):
    H = np.arctan2(-spin[..., 0], -spin[..., 1]) + np.pi / 2.
    H = H % (2 * np.pi) / (2. * np.pi)
    S = np.clip(np.sqrt(spin[..., 0] ** 2 + spin[..., 1] ** 2), 0., 1.)
    I = np.clip((spin[..., 2] + 1.) / 2., 0., 1.)
    r, g, b = np.vectorize(hsv_to_rgb)(H, S, I)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb

def vertical_stripe(height: int = 1260, width: int = 1260, n_stripe: int = 20):
    xx, yy = tf.meshgrid(
        tf.linspace(0., 1., width + 1)[:-1],
        tf.linspace(0., 1., height + 1)[:-1],
    )
    sz = tf.cos(xx * n_stripe * np.pi * 2.)
    sz = tf.where(tf.greater(sz, 0.), 1., -1.)
    s = tf.stack([
        tf.zeros_like(sz),
        tf.zeros_like(sz),
        sz
    ], axis=-1)
    return s

def horizontal_stripe(height: int = 1260, width: int = 1260, n_stripe: int = 20):
    s = vertical_stripe(n_stripe=n_stripe)
    s = tf.transpose(s, (1, 0, 2))
    return s

def diagonal_stripe(height: int = 1260, width: int = 1260, n_stripe: int = 14):
    xx, yy = tf.meshgrid(
        tf.linspace(0., 1., width + 1)[:-1],
        tf.linspace(0., 1., height + 1)[:-1],
    )
    sz = tf.cos((xx + yy) * n_stripe * np.pi * 2.)
    sz = tf.where(tf.greater(sz, 0.), 1., -1.)
    s = tf.stack([
        tf.zeros_like(sz),
        tf.zeros_like(sz),
        sz
    ], axis=-1)
    return s

def off_diagonal_stripe(height: int = 1260, width: int = 1260, n_stripe: int = 14):
    s = diagonal_stripe(n_stripe=n_stripe)[::-1]
    return s

def vertical_dipole(height: int = 1260, width: int = 1260, n_stripe: int = 20):
    xx, yy = np.meshgrid(
        np.linspace(0., 1., width + 1)[:-1],
        np.linspace(0., 1., height + 1)[:-1],
    )
    # left half
    part1_mask = np.float32((0.0 <= xx) & (xx < (0.5 - 0.5 / n_stripe)))
    sz_part1 = np.cos(yy * n_stripe * np.pi * 2.)
    sz_part1 = tf.where(tf.greater(sz_part1, 0.), 1., -1.)

    part2_mask = np.float32(((0.5 - 0.5 / n_stripe) <= xx) & (xx < 0.5))
    sz_part2 = -np.cos((yy - xx) * n_stripe * np.pi * 2.)
    sz_part2[height // 2:] = sz_part2[:height // 2][::-1]
    sz_part2 = tf.where(tf.greater(sz_part2, 0.), 1., -1.)

    sz = sz_part1 * part1_mask + sz_part2 * part2_mask

    # total spin
    sz = tf.concat([
        sz[:, :width // 2],
        -sz[:, :width // 2],
    ], axis=-1)
    s = tf.stack([
        tf.zeros_like(sz),
        tf.zeros_like(sz),
        sz
    ], axis=-1)
    return s

def horizontal_dipole(height: int = 1260, width: int = 1260, n_stripe: int = 20):
    s = vertical_dipole(n_stripe=n_stripe)
    s = tf.transpose(s, (1, 0, 2))
    return s

def quadrupole(height: int = 1260, width: int = 1260, n_stripe: int = 20):
    s = np.array(vertical_stripe(height=height, width=width, n_stripe=n_stripe))
    # disconnection
    box = [
        height // 2 - height // n_stripe // 4,
        width // 2 - width // n_stripe // 4,
        height // 2 + height // n_stripe // 4,
        width // 2 + width // n_stripe // 4 + 1,
    ]
    s[box[0]:box[2], box[1]:box[3], -1] = -1.
    return s

def skyrmion_quadrupole(height: int = 1260, width: int = 1260, n_stripe: int = 20):
    s = np.array(vertical_stripe(height=height, width=width, n_stripe=n_stripe))
    # skyrmion core
    box = [
        height // 2 - height // n_stripe // 4,
        width // 2 - width // n_stripe // 12,
        height // 2 + height // n_stripe // 4,
        width // 2 + width // n_stripe // 12,
    ]
    s[box[0]:box[2], box[1]:box[3]] *= -1
    return s

def get_initial_conditions():
    return tf.stack([
        vertical_stripe(),
        horizontal_stripe(),
        diagonal_stripe(),
        off_diagonal_stripe(),
        vertical_dipole(),
        horizontal_dipole(),
        quadrupole(),
        skyrmion_quadrupole(),
    ])

def greedy_algorithm(
        init_spins,
        iterations: int,
        sub_iterations: int = 100,
        exJ: float = 1.,
        DMB: float = 0.,
        DMN: float = 0.
):
    shape = init_spins.shape[1:3]
    checker_board = (np.indices(shape, dtype=np.float32).sum(axis=0) % 2.)[np.newaxis, ..., np.newaxis]
    spins = tf.constant(init_spins)
    for i in range(iterations):
        old_spin = spins
        heff = get_heff_JDM(spins, exJ=exJ, DMB=DMB, DMN=DMN)
        new_spins = tf.math.l2_normalize(heff, axis=-1)
        spins = spins * checker_board + new_spins * (1. - checker_board)
        new_spins =  tf.math.l2_normalize(spins, axis=-1)
        spins = spins * (1. - checker_board) + new_spins * checker_board
        if (i + 1) % sub_iterations == 0:
            print(f'iteration {i + 1} / {iterations} | Energy: {-tf.reduce_mean(old_spin * heff).numpy() * 3. / 2.}')
    return spins

def get_heff_JDM(spins, exJ: float = 1., DMB: float = 0., DMN: float = 0.):
    padded = pad_2d(spins, pad_i=1, pad_j=1)
    heff_exJ = padded[..., :-2, 1:-1, :] +\
               padded[..., 2:, 1:-1, :] +\
               padded[..., 1:-1, :-2, :] +\
               padded[..., 1:-1, 2:, :]
    heff_DMB = tf.stack([
        -padded[..., :-2, 1:-1, 2] + padded[..., 2:, 1:-1, 2],
        -padded[..., 1:-1, :-2, 2] + padded[..., 1:-1, 2:, 2],
        padded[..., :-2, 1:-1, 0] - padded[..., 2:, 1:-1, 0] + padded[..., 1:-1, :-2, 1] - padded[..., 1:-1, 2:, 1]
    ], axis=-1)
    heff_DMN = tf.stack([
        padded[..., 1:-1, :-2, 2] - padded[..., 1:-1, 2:, 2],
        -padded[..., :-2, 1:-1, 2] + padded[..., 2:, 1:-1, 2],
        padded[..., :-2, 1:-1, 1] - padded[..., 2:, 1:-1, 1] - padded[..., 1:-1, :-2, 0] + padded[..., 1:-1, 2:, 0]
    ], axis=-1)
    heff = heff_exJ * exJ + heff_DMB * DMB + heff_DMN * DMN
    return heff
