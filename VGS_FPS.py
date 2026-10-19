""" VGS -> global skeleton FPS -> density-adaptive XY block FPS -> global FPS."""
import argparse
from pathlib import Path

import numpy as np
import torch


def auto_voxel_size(points, target_count):
    """ AABB estimate + occupancy calibration."""
    local = points - points.min(axis=0)
    extent = local.max(axis=0)
    # V / target is the initial voxel volume; all AABB extents must be positive.
    size = float(np.cbrt(np.prod(extent) / target_count))

    def occupied(edge):
        keys = np.floor(local / edge).astype(np.int64)
        return len(np.unique(keys, axis=0))

    # Bracket a fine grid with enough candidates and a coarse grid with too few.
    for _ in range(24):
        if occupied(size) >= target_count:
            break
        size *= 0.5
    else:
        raise ValueError('Cannot reach the VGS target; check duplicate points or coordinate precision.')
    low, high = size, size * 2
    while occupied(high) >= target_count:
        low, high = high, high * 2

    # Refine eight times, retaining a feasible edge; the ratio is approximate.
    # Occupancy is not strictly monotone for non-nested grids; this is not exact root finding.
    for _ in range(8):
        middle = (low + high) / 2
        if occupied(middle) >= target_count:
            low = middle
        else:
            high = middle
    return low


def voxel_grid_sampling(points, voxel_size):
    """ Keep the original point nearest each centroid."""
    local = points - points.min(axis=0)
    keys = np.floor(local / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    centroids = np.column_stack([
        np.bincount(inverse, weights=local[:, axis]) / counts for axis in range(3)
    ])
    delta = local - centroids[inverse]
    distance = np.einsum('ij,ij->i', delta, delta)
    order = np.lexsort((np.arange(len(points)), distance, inverse))
    return order[np.r_[0, np.cumsum(counts)[:-1]]]


@torch.no_grad()
def farthest_point_sampling(points, n_samples, seed=0, chunk_size=100000):
    """ Greedy CUDA FPS returning input row indices."""
    if n_samples == len(points):
        return np.arange(len(points))
    #  Translate in float64 before converting to float32.
    local = np.asarray(points, dtype=np.float64) - points.min(axis=0)
    xyz = torch.as_tensor(local, dtype=torch.float32, device='cuda')
    selected = torch.empty(n_samples, dtype=torch.long, device='cuda')
    selected[0] = int(np.random.default_rng(seed).integers(len(points)))
    minimum = torch.full((len(points),), float('inf'), device='cuda')
    for i in range(1, n_samples):
        center = xyz[selected[i - 1]]
        # Batch distance updates only; this does not partition the FPS search domain.
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            distance = ((xyz[start:stop] - center) ** 2).sum(dim=1)
            minimum[start:stop] = torch.minimum(minimum[start:stop], distance)
        minimum[selected[i - 1]] = -float('inf')
        selected[i] = minimum.argmax()
    return selected.cpu().numpy()


def spatial_blocks(points, block_size, origin):
    """ Fixed XY cells spanning the full Z range."""
    keys = np.floor((points[:, :2] - origin[:2]) / block_size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    return np.split(order, np.cumsum(counts)[:-1])


def allocate_block_quotas(counts, total, density_power=0.5):
    """ Allocate integer quotas by density."""
    # Equal-area XY cells: rho_i = N_i / (Bx * By); weights scale as N_i ** alpha.
    weights = counts.astype(np.float64) ** density_power
    quotas = np.zeros(len(counts), dtype=np.int64)
    if total >= len(counts):
        quotas[:] = 1
    while quotas.sum() < total:
        active = np.flatnonzero(quotas < counts)
        remaining = total - quotas.sum()
        shares = remaining * weights[active] / weights[active].sum()
        addition = np.minimum(np.floor(shares).astype(np.int64), counts[active] - quotas[active])
        if addition.sum():
            quotas[active] += addition
        else:
            # Assign the remaining integer slots by largest remainder.
            order = np.argsort(-(shares - np.floor(shares)), kind='stable')
            quotas[active[order[:remaining]]] += 1
    return quotas


def vgs_fps_sampling(points, sample_ratio=1 / 40, *, block_size,
                     skeleton_ratio=0.001, merge_factor=2.0, density_power=0.5,
                     seed=0, chunk_size=100000):
    """ Return original row indices and coordinates."""
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA-enabled PyTorch environment is required.')
    points = np.asarray(points, dtype=np.float64)
    block_size = np.asarray(block_size)
    n_samples = int(len(points) * sample_ratio)
    # Parameter bounds only; the 1/5 VGS rule requires a final retention ratio below 1/5.
    if not (0 < sample_ratio < 0.2 and n_samples >= 1 and
            block_size.shape == (2,) and np.all(block_size > 0) and
            0 < skeleton_ratio <= 1 and merge_factor > 1 and
            0 <= density_power <= 1 and chunk_size > 0):
        raise ValueError('Require 0 < sample_ratio < 0.2, at least one output point, and valid parameters.')
    fps_options = dict(seed=seed, chunk_size=chunk_size)

    # Stage 1: target N/10 for r <= 1/20, otherwise N/5; estimate and calibrate voxels.
    vgs_ratio = 0.1 if sample_ratio <= 0.05 else 0.2
    vgs_target = int(np.ceil(len(points) * vgs_ratio))
    print(f'Stage 1: VGS target={vgs_target:,}; calibrating voxel size...', flush=True)
    voxel_size = auto_voxel_size(points, vgs_target)
    candidate_ids = voxel_grid_sampling(points, voxel_size)
    candidates = points[candidate_ids]
    print(f'Voxel size={voxel_size:.6g}; candidates={len(candidates):,} '
          f'({len(candidates) / len(points):.4%})', flush=True)

    # Stage 2: skeleton count = ceil(final count * skeleton_ratio), relative to the final count.
    skeleton_count = max(1, int(np.ceil(n_samples * skeleton_ratio)))
    print(f'Stage 2: skeleton FPS -> {skeleton_count:,}', flush=True)
    skeleton = farthest_point_sampling(candidates, skeleton_count, **fps_options)
    mask = np.ones(len(candidates), dtype=bool)
    mask[skeleton] = False
    remaining = np.flatnonzero(mask)

    # Stage 3: partition remaining candidates in XY; allocate local FPS quotas by their density.
    blocks = spatial_blocks(candidates[remaining], block_size, points.min(axis=0))
    merged_count = min(len(candidates), max(n_samples + 1, int(np.ceil(n_samples * merge_factor))))
    counts = np.array([len(block) for block in blocks], dtype=np.int64)
    quotas = allocate_block_quotas(counts, merged_count - skeleton_count, density_power)
    print(f'Stage 3: {len(blocks):,} XY blocks; local FPS -> {quotas.sum():,}', flush=True)
    sampled_blocks = []
    for block, quota in zip(blocks, quotas):
        if quota:
            pool = remaining[block]
            sampled_blocks.append(pool[farthest_point_sampling(candidates[pool], int(quota), **fps_options)])

    # Stage 4: merge skeleton/local samples; global FPS returns exactly floor(N * r) rows.
    merged = np.concatenate([skeleton] + sampled_blocks)
    print(f'Stage 4: global FPS {len(merged):,} -> {n_samples:,}', flush=True)
    final = farthest_point_sampling(candidates[merged], n_samples, **fps_options)
    indices = candidate_ids[merged[final]]
    return indices, points[indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--sample-ratio', type=float, default=1 / 40,
                        help='Final retained fraction of input rows, e.g. 0.025 = 1/40 (default).')
    parser.add_argument('--block-size', type=float, nargs=2, required=True, metavar=('BX', 'BY'),
                        help='XY cell widths in coordinate units, e.g. 100 100 for metre coordinates.')
    parser.add_argument('--skeleton-ratio', type=float, default=0.001)
    parser.add_argument('--merge-factor', type=float, default=2.0)
    parser.add_argument('--density-power', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--chunk-size', type=int, default=100000,
                        help='CUDA distance-update batch size; unrelated to XY spatial cells.')
    args = parser.parse_args()
    output = Path(args.output_file)
    indices_file = Path(str(output) + '.indices.npy')
    if output.resolve() == Path(args.input_file).resolve() or output.exists() or indices_file.exists():
        parser.error('Please use a new output path.')

    # Input: column names, point count, comma-separated rows; XYZ in the first three columns.
    with open(args.input_file, encoding='utf-8-sig') as f:
        header = f.readline().strip()
        f.readline()
        data = np.loadtxt(f, delimiter=',', dtype=np.float64, ndmin=2)
    indices, _ = vgs_fps_sampling(
        data[:, :3], args.sample_ratio, block_size=args.block_size,
        skeleton_ratio=args.skeleton_ratio, merge_factor=args.merge_factor,
        density_power=args.density_power, seed=args.seed, chunk_size=args.chunk_size)
    # Save original rows, preserving attributes and labels.
    with output.open('w', encoding='utf-8') as f:
        f.write(f'{header}\n{len(indices)}\n')
        np.savetxt(f, data[indices], delimiter=',', fmt='%.17g')
    np.save(indices_file, indices)
    print(f'Done: {len(data):,} -> {len(indices):,}; {output}')


if __name__ == '__main__':
    main()
