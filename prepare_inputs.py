"""Convert the frozen research inputs to one portable NPZ per target cohort.

The source directory is read-only. No preprocessing is refitted and no patients
are re-split. Run this on trusted local inputs; none are distributed here.
"""
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import io_utils as R

CANCERS = ['luad', 'lgg', 'kirc', 'brca', 'lihc']


def convert(root, cancer):
    paths = []

    def read(relative):
        path = root / relative
        paths.append(path)
        with np.load(path, allow_pickle=True) as f:
            return dict(f)

    if cancer == 'luad':
        d = read('data/cohort/cohort.npz')
        cl = read('data/extra.npz')
        wide = read('data/extra2.npz')
        prior = read('data/prior15.npz')
        coords = read('pancan19.npz')
        ids = np.arange(len(d['time']))
        np.testing.assert_array_equal(cl['patients'], d['patients'])
        np.testing.assert_array_equal(wide['patients'], d['patients'])
        full = wide['rna_full'][:, coords['gene_idx']]
        broad = coords['genes'].astype(str)
    else:
        d = read(f'data/{cancer}/cohort.npz')
        cl = read(f'data/{cancer}/clin.npz')
        wide = read(f'data/{cancer}/rnapan.npz')
        prior = read(f'data/{cancer}/prior.npz')
        ids = np.flatnonzero(np.isfinite(d['time']) & (d['time'] > 0))
        np.testing.assert_array_equal(cl['patients'][ids], d['patients'][ids])
        np.testing.assert_array_equal(wide['samples'][ids], d['samples'][ids])
        assert str(prior['cancer']) == cancer and (prior['prog_n'] == 3).all()
        full = wide['rna'][ids]
        broad = wide['genes'].astype(str)
    np.testing.assert_array_equal(prior['genes'], d['genes'])
    data = dict(x=d['x'][ids, :1, :], clinical=cl['clin'][ids].astype(float),
                full_rna=full, broad_genes=broad, genes=d['genes'].astype(str),
                prior=prior['prog'].astype(float), patients=d['patients'][ids].astype(str),
                time=d['time'][ids], event=d['event'][ids])
    for split in range(1, 11):
        if cancer == 'luad':
            prep = read(f'data/cohort/split_{split:02}/preprocessing.npz')
            parts = {p: prep[p] for p in ['train', 'val', 'test']}
        else:
            tr, rest = train_test_split(np.arange(len(ids)), test_size=.2,
                                       stratify=data['event'], random_state=split)
            va, te = train_test_split(rest, test_size=.5,
                                     stratify=data['event'][rest], random_state=100 + split)
            parts = dict(train=tr, val=va, test=te)
        np.testing.assert_array_equal(np.sort(np.concatenate(list(parts.values()))), np.arange(len(ids)))
        data.update({f'{p}_{split:02}': value for p, value in parts.items()})
    return data, paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, default=Path('data'))
    p.add_argument('--cancers', nargs='+', choices=CANCERS, default=CANCERS)
    args = p.parse_args()
    destinations = [args.out_dir / f'{c}.npz' for c in args.cancers]
    if any(path.exists() for path in destinations):
        p.error('A destination cohort already exists; use a new output directory')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import pipeline as P
    P.DATA_DIR = args.out_dir.resolve()
    common_genes = None
    for c, dest in zip(args.cancers, destinations):
        data, paths = convert(args.legacy_root.resolve(), c)
        if common_genes is not None:
            np.testing.assert_array_equal(common_genes, data['broad_genes'])
        common_genes = data['broad_genes']
        np.savez_compressed(dest, **data)
        loaded = P.load_data(c)
        assert len(loaded['time']) == len(data['time'])
        R.write_json(dest.with_suffix('.manifest.json'), dict(
            cancer=c, patients=len(data['time']), sha256=R.sha(dest),
            source_sha256={str(path.relative_to(args.legacy_root.resolve())): R.sha(path) for path in paths}))
        print(c, len(data['time']), 'patients:', dest, flush=True)


if __name__ == '__main__':
    main()
