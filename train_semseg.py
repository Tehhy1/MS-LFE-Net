"""矿区语义分割实验训练 / Mine semantic-segmentation experiments (CUDA).

Paper defaults: Adam, batch=8, epochs=100, lr=0.0005, inverse-frequency WCE.
Inputs: prebuilt 4096-point blocks; split the scene in X (80/20) BEFORE
building 120 m windows with 80 m stride. Coordinates remain in metres.
"""
import argparse
import copy
import csv
import importlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CLASSES = ('bench', 'slope', 'benchline')
BASE_DIR = Path(__file__).resolve().parent


class PointBlocks(Dataset):
    def __init__(self, folder, split):
        folder = Path(folder)
        self.points = np.load(folder / f'blocks_points_{split}.npy', mmap_mode='r')
        self.labels = np.load(folder / f'blocks_labels_{split}.npy')
        self.indices = np.load(folder / f'blocks_indices_{split}.npy')

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        points = self.points[i].astype(np.float64)
        # Translate only; preserve metre units for the SA radii.
        points[:, :3] -= points[:1, :3]
        return points.astype(np.float32), self.labels[i].astype(np.int64), self.indices[i].astype(np.int64)


def class_weights(labels):
    # inverse class frequencies of UNIQUE training points only.
    counts = np.bincount(labels, minlength=3)
    return torch.tensor(counts.sum() / counts, dtype=torch.float32)


def metrics_from_confusion(matrix):
    """行=真值，列=预测；公式(16)-(20)。 / Rows=true, columns=predicted; Eqs. 16-20."""
    matrix = np.asarray(matrix, dtype=np.int64)
    tp = np.diag(matrix).astype(float)
    support, predicted = matrix.sum(1), matrix.sum(0)
    fp, fn = predicted - tp, support - tp
    tn = matrix.sum() - tp - fp - fn

    def divide(a, b):
        # Define zero-denominator class ratios as 0; always average over three classes.
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    iou, precision, recall = divide(tp, tp + fp + fn), divide(tp, predicted), divide(tp, support)
    proportions = divide(support, support.sum())
    oa = float(divide(tp.sum(), support.sum()))
    return dict(oa=oa, mean_iou=float(iou.mean()),
                macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
                weighted_precision=float(precision @ proportions), weighted_recall=float(recall @ proportions),
                micro_precision=oa, micro_recall=oa,
                iou_per_class=iou.tolist(), precision_per_class=precision.tolist(),
                recall_per_class=recall.tolist(), support=support.tolist(),
                tp=tp.astype(int).tolist(), fp=fp.astype(int).tolist(),
                fn=fn.astype(int).tolist(), tn=tn.astype(int).tolist(),
                confusion_matrix=matrix.tolist())


def print_metrics(log, title, report, show_matrix=False):
    log.info('%s | OA=%.2f%% | mIoU=%.2f%% | Precision(macro)=%.2f%% | Recall(macro)=%.2f%%',
             title, 100 * report['oa'], 100 * report['mean_iou'],
             100 * report['macro_precision'], 100 * report['macro_recall'])
    for i, name in enumerate(CLASSES):
        log.info('  %-9s IoU=%6.2f%%  Precision=%6.2f%%  Recall=%6.2f%%  support=%d',
                 name, 100 * report['iou_per_class'][i], 100 * report['precision_per_class'][i],
                 100 * report['recall_per_class'][i], report['support'][i])
    if show_matrix:
        log.info('Precision(weighted)=%.2f%% | Recall(weighted)=%.2f%% | micro P/R=OA',
                 100 * report['weighted_precision'], 100 * report['weighted_recall'])
        log.info('Confusion matrix: rows=true, columns=predicted; class order=%s\n%s',
                 CLASSES, np.asarray(report['confusion_matrix']))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False),
                          encoding='utf-8')


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with Path(path).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def metric_row(prefix, report):
    row = {prefix + key: report[key] for key in
           ('oa', 'mean_iou', 'macro_precision', 'macro_recall', 'weighted_precision', 'weighted_recall')}
    for i, name in enumerate(CLASSES):
        for metric in ('iou', 'precision', 'recall'):
            row[f'{prefix}{name}_{metric}'] = report[f'{metric}_per_class'][i]
    return row


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state(generator):
    np_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(np_state[0], np_state[1].tolist(), *np_state[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state['python'])
    name, keys, position, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    generator.set_state(state['loader'].cpu())


def make_loader(dataset, args, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=True, drop_last=False, worker_init_fn=seed_worker, generator=generator)


def train_epoch(model, criterion, optimizer, loader, weights, log, print_every):
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    numerator, denominator = 0., 0.
    confusion = torch.zeros((3, 3), dtype=torch.int64, device='cuda')
    for step, (points, target, _) in enumerate(loader, 1):
        points = points.cuda(non_blocking=True).transpose(1, 2)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        result = model(points)
        # Paper WCE only; the CE option supports the without-WCE ablation.
        loss = criterion(result[0].reshape(-1, 3), target.reshape(-1), result[1], weights)
        loss.backward()
        optimizer.step()
        weight_sum = weights[target].sum().item() if weights is not None else target.numel()
        numerator += loss.item() * weight_sum
        denominator += weight_sum
        predicted = result[0].detach().argmax(-1)
        confusion += torch.bincount((target * 3 + predicted).reshape(-1), minlength=9).reshape(3, 3)
        if print_every and (step % print_every == 0 or step == len(loader)):
            log.info('  Batch %d/%d | loss=%.6f', step, len(loader), loss.item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    # Training metrics count block occurrences and are monitoring metrics, not scene test scores.
    report = metrics_from_confusion(confusion.cpu().numpy())
    report.update(loss=numerator / denominator, train_seconds=elapsed, train_minutes=elapsed / 60,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)
    return report


@torch.no_grad()
def evaluate_scene(model, criterion, loader, raw_labels, expected_ids, weights):
    model.eval()
    votes = np.zeros((len(expected_ids), 3), dtype=np.int64)
    numerator, denominator = 0., 0.
    torch.cuda.synchronize()
    start = time.perf_counter()
    for points, target, indices in loader:
        result = model(points.cuda(non_blocking=True).transpose(1, 2))
        target = target.cuda(non_blocking=True)
        loss = criterion(result[0].reshape(-1, 3), target.reshape(-1), result[1], weights)
        weight_sum = weights[target].sum().item() if weights is not None else target.numel()
        numerator += loss.item() * weight_sum
        denominator += weight_sum
        predictions = result[0].argmax(-1).cpu().numpy()
        # Vote by original row ID; collapse repeated padding rows within each block first.
        for ids, pred in zip(indices.numpy(), predictions):
            unique, inverse = np.unique(ids, return_inverse=True)
            within = np.zeros((len(unique), 3), dtype=np.int64)
            np.add.at(within, (inverse, pred), 1)
            votes[np.searchsorted(expected_ids, unique), within.argmax(1)] += 1
    covered = votes.sum(1) > 0
    predictions = np.full(len(expected_ids), -1, dtype=np.int64)
    predictions[covered] = votes[covered].argmax(1)
    # count each original point once after fusion; uncovered predictions stay -1.
    truth = raw_labels[expected_ids[covered]]
    matrix = np.bincount(truth * 3 + predictions[covered], minlength=9).reshape(3, 3)
    report = metrics_from_confusion(matrix)
    torch.cuda.synchronize()
    report.update(block_loss=numerator / denominator, evaluation_seconds=time.perf_counter() - start,
                  expected_points=len(expected_ids), covered_points=int(covered.sum()),
                  coverage=float(covered.mean()),
                  evaluation_scope='full_requested_region' if covered.all() else 'covered_subset_only')
    return report, predictions


def save_evaluation(output, split, report, predictions, ids):
    write_json(output / f'{split}_metrics.json', report)
    write_csv(output / f'{split}_metrics.csv', [metric_row('', report) | {
        'coverage': report['coverage'], 'evaluation_scope': report['evaluation_scope']}])
    rows = [dict(class_id=i, class_name=name, **{key: report[key][i] for key in
            ('support', 'tp', 'fp', 'fn', 'tn')}, IoU=report['iou_per_class'][i],
                 Precision=report['precision_per_class'][i], Recall=report['recall_per_class'][i])
            for i, name in enumerate(CLASSES)]
    write_csv(output / f'{split}_per_class.csv', rows)
    matrix = np.asarray(report['confusion_matrix'])
    write_csv(output / f'{split}_confusion.csv', [dict(true_class=name, **dict(zip(CLASSES, row)))
                                               for name, row in zip(CLASSES, matrix)])
    np.save(output / f'{split}_confusion.npy', matrix)
    np.save(output / f'{split}_indices.npy', ids)
    np.save(output / f'{split}_predictions.npy', predictions)


def configure_schedule(model, optimizer, args, epoch):
    # Retain the existing LR/BN schedules; their decay details are implementation settings.
    lr = max(args.learning_rate * args.lr_decay ** (epoch // args.step_size), 1e-5)
    momentum = max(0.1 * 0.5 ** (epoch // args.step_size), 0.01)
    for group in optimizer.param_groups:
        group['lr'] = lr
    for layer in model.modules():
        if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            layer.momentum = momentum
    return lr, momentum


def run_experiment(args):
    seed_everything(args.seed)
    output = args.log_dir
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    log = logging.getLogger(f'mine.{output.resolve()}')
    log.setLevel(logging.INFO)
    log.propagate = False
    log.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(output / 'training.log', encoding='utf-8')):
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        log.addHandler(handler)
    try:
        return _run(args, output, checkpoints, log)
    finally:
        for handler in log.handlers[:]:
            handler.close()
            log.removeHandler(handler)


def _run(args, output, checkpoints, log):
    state = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if state is not None and args.eval_only:
        if state.get('config', {}).get('model', args.model) != args.model:
            raise ValueError('Use the checkpoint model name with --model.')
        args.loss = state.get('config', {}).get('loss', args.loss)
    module = importlib.import_module('models.' + args.model)
    model, criterion = module.get_model(3).cuda(), module.get_loss().cuda()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    stamp = time.strftime('%Y%m%d_%H%M%S')
    metadata = dict(config=config, classes=CLASSES, python=platform.python_version(), numpy=np.__version__,
                    pytorch=str(torch.__version__), cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), platform=platform.platform(),
                    model_parameters=sum(p.numel() for p in model.parameters()),
                    metric_convention='fixed-three-class macro; zero denominator -> 0; JSON/CSV in [0,1]')
    write_json(output / f'config_{stamp}.json', metadata)
    source = output / 'source' / stamp
    (source / 'models').mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), source / 'train_semseg.py')
    for file in (BASE_DIR / 'models').glob('*.py'):
        shutil.copy2(file, source / 'models' / file.name)
    log.info('Model=%s | seed=%d | loss=%s | GPU=%s | parameters=%s',
             args.model, args.seed, args.loss, metadata['gpu'], f"{metadata['model_parameters']:,}")
    raw_points = np.load(args.data_root / 'raw_points.npy', mmap_mode='r')
    raw_labels = np.load(args.data_root / 'raw_labels.npy').astype(np.int64)
    split_x = float(raw_points[:, 0].min() + 0.8 * np.ptp(raw_points[:, 0]))
    if state is not None:
        model.load_state_dict(state['model_state_dict'])
    history, weights = [], None
    selected_by, selected_epoch = 'specified_checkpoint', state['epoch'] + 1 if state else None
    selected_path = args.resume
    if args.eval_only:
        # Use training weights stored in the checkpoint; never estimate weights from test labels.
        saved_weights = state.get('class_weights')
        if args.loss == 'wce':
            if saved_weights is None:
                train = PointBlocks(args.data_root / 'train', 'train')
                saved_weights = class_weights(raw_labels[np.unique(train.indices)])
            weights = saved_weights.cuda()
    else:
        train = PointBlocks(args.data_root / 'train', 'train')
        train_ids = np.unique(train.indices)
        # Enforce experiment separation without modifying FPS/grouping or adding geometric fallbacks.
        if np.any(raw_points[train_ids, 0] >= split_x):
            raise ValueError('Training blocks cross the X-axis 80/20 split. Rebuild blocks after splitting.')
        weights = class_weights(raw_labels[train_ids]).cuda() if args.loss == 'wce' else None
        log.info('Train blocks=%d | points/block=%d | unique train points=%d | class counts=%s | weights=%s',
                 len(train), train.points.shape[1], len(train_ids),
                 np.bincount(raw_labels[train_ids], minlength=3).tolist(),
                 weights.cpu().tolist() if weights is not None else 'uniform CE')
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = make_loader(train, args, shuffle=True, generator=generator)
        val_loader, val_ids = None, None
        if args.val_dir:
            val = PointBlocks(args.val_dir, 'val')
            val_ids = np.unique(val.indices)
            if np.intersect1d(train_ids, val_ids).size or np.any(raw_points[val_ids, 0] >= split_x):
                raise ValueError('Validation must be disjoint from training and within the training 80% region.')
            val_loader = make_loader(val, args, generator=torch.Generator().manual_seed(args.seed + 1))
        selected_by = 'validation_miou' if val_loader is not None else 'final_epoch'
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
        start_epoch, best_miou, best_epoch, best_model = 0, -1., None, None
        if state is not None:
            # Restore optimizer, RNG and history; --epoch is the total number of epochs.
            for key in ('model', 'loss', 'batch_size', 'learning_rate', 'decay_rate', 'step_size', 'lr_decay', 'seed'):
                if state.get('config', {}).get(key, config[key]) != config[key]:
                    raise ValueError(f'Resume setting differs from checkpoint: {key}')
            if state.get('selected_by', selected_by) != selected_by:
                raise ValueError('Keep the same validation/selection protocol when resuming.')
            optimizer.load_state_dict(state['optimizer_state_dict'])
            start_epoch = state['epoch'] + 1
            history = state.get('history', [])
            best_miou, best_epoch = state.get('best_miou', -1.), state.get('best_epoch')
            best_model = state.get('best_model_state_dict')
            if 'rng_state' in state:
                restore_rng(state['rng_state'], generator)
            else:
                log.info('Legacy checkpoint has no RNG/history state; exact continuation is unavailable.')
            log.info('Resuming after epoch %d; target total epochs=%d', start_epoch, args.epoch)
        for epoch in range(start_epoch, args.epoch):
            lr, momentum = configure_schedule(model, optimizer, args, epoch)
            train_report = train_epoch(model, criterion, optimizer, train_loader, weights, log, args.print_every)
            print_metrics(log, f'Epoch {epoch + 1}/{args.epoch} TRAIN (block occurrences)', train_report)
            log.info('loss=%.6f | lr=%.8f | train time=%.3f min | peak allocated/reserved=%.3f/%.3f GiB',
                     train_report['loss'], lr, train_report['train_minutes'],
                     train_report['peak_allocated_gib'], train_report['peak_reserved_gib'])
            row = dict(epoch=epoch + 1, learning_rate=lr, bn_momentum=momentum,
                       train_loss=train_report['loss'], train_minutes=train_report['train_minutes'],
                       peak_allocated_gib=train_report['peak_allocated_gib'],
                       peak_reserved_gib=train_report['peak_reserved_gib'], **metric_row('train_', train_report))
            if val_loader is not None:
                validation, _ = evaluate_scene(model, criterion, val_loader, raw_labels, val_ids, weights)
                print_metrics(log, 'VALIDATION (unique indexed points)', validation)
                row.update(val_block_loss=validation['block_loss'],
                           val_seconds=validation['evaluation_seconds'], **metric_row('val_', validation))
                if validation['mean_iou'] > best_miou:
                    best_miou, best_epoch = validation['mean_iou'], epoch + 1
                    best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(dict(epoch=epoch, model_state_dict=best_model,
                                    class_weights=weights.cpu() if weights is not None else None,
                                    config=config, selected_by=selected_by), checkpoints / 'best_model.pth')
            history.append(row)
            write_csv(output / 'history.csv', history)
            state = dict(epoch=epoch, model_state_dict=model.state_dict(),
                         optimizer_state_dict=optimizer.state_dict(), rng_state=rng_state(generator),
                         class_weights=weights.cpu() if weights is not None else None,
                         history=history, config=config, selected_by=selected_by,
                         best_miou=best_miou, best_epoch=best_epoch, best_model_state_dict=best_model)
            torch.save(state, checkpoints / 'last_model.pth')
        if val_loader is not None:
            model.load_state_dict(best_model)
            selected_epoch = best_epoch
            selected_path = checkpoints / 'best_model.pth'
            # Best checkpoint is for inference; resume training from last_model.pth.
            torch.save(dict(epoch=best_epoch - 1, model_state_dict=best_model,
                            class_weights=weights.cpu() if weights is not None else None,
                            config=config, selected_by=selected_by), selected_path)
        else:
            selected_epoch = state['epoch'] + 1
            selected_path = checkpoints / 'last_model.pth' if args.epoch > start_epoch else args.resume
        summary = dict(selected_by=selected_by, selected_epoch=selected_epoch, seed=args.seed,
                       measured_epochs=len(history),
                       mean_train_minutes=float(np.mean([r['train_minutes'] for r in history])) if history else None,
                       peak_allocated_gib=max((r['peak_allocated_gib'] for r in history), default=None),
                       peak_reserved_gib=max((r['peak_reserved_gib'] for r in history), default=None))
        write_json(output / 'training_summary.json', summary)
        log.info('Training summary: %s', json.dumps(summary))
    if args.skip_test:
        return None
    # Evaluate the test set only after training and checkpoint selection have finished.
    test = PointBlocks(args.data_root / 'test', 'test')
    if np.any(raw_points[np.unique(test.indices), 0] < split_x):
        raise ValueError('Test blocks cross into the training 80% region.')
    test_ids = np.flatnonzero(raw_points[:, 0] >= split_x)
    loader = make_loader(test, args, generator=torch.Generator().manual_seed(args.seed + 2))
    report, predictions = evaluate_scene(model, criterion, loader, raw_labels, test_ids, weights)
    report.update(seed=args.seed, classes=list(CLASSES), selected_by=selected_by,
                  selected_epoch=selected_epoch, checkpoint=str(selected_path), loss_kind=args.loss,
                  metric_units='fractions in [0,1]', confusion_orientation='rows=true, columns=predicted')
    save_evaluation(output, 'test', report, predictions, test_ids)
    log.info('TEST coverage=%d/%d (%.2f%%); scope=%s; checkpoint=%s', report['covered_points'],
             report['expected_points'], 100 * report['coverage'], report['evaluation_scope'], selected_path)
    print_metrics(log, 'FINAL TEST', report, show_matrix=True)
    if report['coverage'] < 1:
        log.info('Coverage is incomplete: these are covered-subset scores, not full-scene paper results. '
                 'Generate inference blocks covering every test point; uncovered predictions are -1.')
    log.info('Evaluation time (loading + forward + voting + metrics)=%.3f min', report['evaluation_seconds'] / 60)
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', type=Path, required=True)
    p.add_argument('--log_dir', type=Path, default=Path('log/sem_seg') / time.strftime('%Y-%m-%d_%H-%M-%S'))
    p.add_argument('--model', default='pointnet2_sem_seg_msg')
    p.add_argument('--val_dir', type=Path, help='Optional disjoint validation blocks inside the training 80%%.')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epoch', type=int, default=100)
    p.add_argument('--learning_rate', type=float, default=0.0005)
    p.add_argument('--decay_rate', type=float, default=5e-4)
    p.add_argument('--step_size', type=int, default=20)
    p.add_argument('--lr_decay', type=float, default=0.9)
    p.add_argument('--loss', choices=('wce', 'ce'), default='wce')
    p.add_argument('--gpu', default='0')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seeds', type=int, nargs='+', help='Independent repetitions, e.g. --seeds 0 1 2.')
    p.add_argument('--resume', type=Path, help='Resume from last_model.pth; --epoch specifies total epochs.')
    p.add_argument('--eval_only', action='store_true', help='Only evaluate the checkpoint supplied with --resume.')
    p.add_argument('--skip_test', action='store_true', help='Training/validation only; leave the test set unused.')
    p.add_argument('--print_every', type=int, default=20)
    args = p.parse_args(argv)
    if args.eval_only and (args.resume is None or args.skip_test):
        p.error('--eval_only requires --resume and cannot use --skip_test.')
    if args.seeds and (args.resume or len(set(args.seeds)) != len(args.seeds)):
        p.error('--seeds requires distinct seeds and starts new runs; resume runs individually.')
    if min(args.batch_size, args.epoch, args.step_size) <= 0 or args.workers < 0:
        p.error('batch_size, epoch and step_size must be positive; workers must be nonnegative.')
    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Run this script on the experiment workstation.')
    if args.log_dir.exists() and any(args.log_dir.iterdir()) and not args.resume:
        raise ValueError('Use a new --log_dir for a new experiment, or --resume to continue one.')
    if not args.seeds:
        return run_experiment(args)
    reports = []
    for seed in args.seeds:
        run_args = copy.copy(args)
        run_args.seed, run_args.seeds = seed, None
        run_args.log_dir = args.log_dir / f'seed_{seed}'
        report = run_experiment(run_args)
        if report is not None:
            reports.append(report)
    if reports:
        # independent-run mean/std; record population and sample conventions.
        summary = dict(seeds=args.seeds, runs=len(reports),
                       all_runs_full_coverage=all(r['coverage'] == 1 for r in reports),
                       metric_units='fractions; multiply by 100 for percent or percentage points')
        rows = [dict(seed=r['seed'], coverage=r['coverage'], **metric_row('', r)) for r in reports]
        for key in metric_row('', reports[0]):
            values = np.array([metric_row('', r)[key] for r in reports])
            summary[key] = dict(mean=float(values.mean()), std_population=float(values.std(ddof=0)),
                                std_sample=float(values.std(ddof=1)) if len(values) > 1 else None)
        write_json(args.log_dir / 'repeated_summary.json', summary)
        write_csv(args.log_dir / 'repeated_runs.csv', rows)
        miou = summary['mean_iou']
        print(f"Repeated mIoU: {miou['mean'] * 100:.2f}% +/- "
              f"{miou['std_population'] * 100:.2f} percentage points (ddof=0); "
              f"full coverage in all runs: {summary['all_runs_full_coverage']}", flush=True)
    return reports


if __name__ == '__main__':
    main(parse_args())
