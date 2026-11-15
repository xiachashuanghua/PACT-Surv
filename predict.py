"""Predict from a trusted frozen PACT-Surv checkpoint; optionally export test IG."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import io_utils as R
import pipeline as P
import train as U


def load_checkpoint(path, expected_sha256=None):
    selection_path = path.parent / 'selection.json'
    if expected_sha256 is None:
        if not selection_path.is_file():
            raise ValueError('Provide the trusted checkpoint SHA-256 or its selection.json')
        expected_sha256 = json.loads(selection_path.read_text())['model_sha256']
    if R.sha(path) != expected_sha256:
        raise ValueError('Checkpoint SHA-256 mismatch')
    # Pickle executes code: a matching digest only helps when its source is trusted.
    return pickle.loads(path.read_bytes())


def selected_sources(source_dir, cancer, state):
    if state['ft'] is None:
        return []
    folder = source_dir / cancer / f"p0.5_wd{state['source_wd']}_lr1e-05"
    nets = []
    for seed in range(5):
        path = folder / f's{seed}.npz'
        key = f"{state['source_wd']}/{seed}"
        expected = state['signature']['source'][key]
        if R.sha(path) != expected:
            raise ValueError(f'Source-network hash mismatch for seed {seed}')
        with np.load(path) as f:
            nets.append(dict(f))
    return nets


def decompose(d, ids, state, nets):
    import interpret as I
    s = state['linear']
    q = d['clinical'].shape[1]
    clinic = (np.where(np.isnan(d['clinical'][ids]), s['clinical_median'], d['clinical'][ids]) - s['clinical_mean']) / s['clinical_sd']
    raw = d['x'][ids, 0]
    rna = (np.where(np.isnan(raw), state['rna_impute'], raw) - s['rna_mean']) / s['rna_sd']
    clinical = clinic @ s['coef'][:q]
    sparse = rna[:, s['idx'][:s['k']]] @ s['coef'][q:]
    prep = state['network_prep']
    raw = d['full_rna'][ids][:, d['gene_idx']]
    z = np.clip((np.where(np.isnan(raw), prep['impute'], raw) - prep['mu']) / prep['sd'], -5, 5).astype(np.float32)
    ig, correction, reference, audit = I.neural_ig(z, nets, state['ft'])
    direct = U.predict(d, ids, state, nets)
    reconstructed = clinical + sparse + reference + ig.sum(1)
    I.require_close(clinical + sparse + correction, direct, 'risk components', 1e-6, 1e-6)
    I.require_close(reconstructed, direct, 'IG reconstruction', I.ARGS.completeness_atol, I.ARGS.completeness_rtol)
    return dict(clinical=clinical, sparse_rna=sparse, reference=reference,
                transfer_change=correction - reference, gene_ig=ig, direct=direct,
                reconstructed=reconstructed, residual=direct - reconstructed), audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cancer', required=True, choices=U.CANCERS)
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--source-dir', type=Path, default=Path('sources'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--sha256')
    p.add_argument('--split', type=int, choices=range(1, 11), required=True)
    p.add_argument('--part', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ig', action='store_true', help='Export zero-reference IG for test patients')
    args = p.parse_args()
    targets = [args.out, args.out.with_suffix('.metrics.json')]
    if args.ig:
        if args.part != 'test':
            p.error('Patient-level IG export is restricted to the outer test set')
        targets.extend([args.out.with_suffix('.ig.npz'), args.out.with_suffix('.ig_audit.json')])
    if any(path.exists() for path in targets):
        p.error('Output exists; choose a new output path')
    P.DATA_DIR = args.data_dir.resolve()
    U.DEV = torch.device('cpu')
    state = load_checkpoint(args.checkpoint, args.sha256)
    d = P.load_data(args.cancer)
    _, expected_ids = P.split_data(d, args.split)
    for part in ['train', 'val', 'test']:
        np.testing.assert_array_equal(state['ids'][part], expected_ids[part])
    nets = selected_sources(args.source_dir, args.cancer, state)
    ids = state['ids'][args.part]
    risk = U.predict(d, ids, state, nets)
    import metrics
    report = dict(cancer=args.cancer, outer_split_id=args.split, partition=args.part,
                  checkpoint_sha256=R.sha(args.checkpoint), n_records=len(ids),
                  c_index=U.score(risk, d, ids),
                  pooled_auc=metrics.pooled_auc(risk, d['time'][ids], d['event'][ids]),
                  horizons=metrics.horizon_auc(risk, d['time'][ids], d['event'][ids]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.ig:
        components, audits = decompose(d, ids, state, nets)
        np.savez_compressed(args.out.with_suffix('.ig.npz'), **components,
                            patient_id=d['patients'][ids], cancer=args.cancer,
                            outer_split_id=args.split, test_status=True, genes=d['broad_genes'])
        R.write_json(args.out.with_suffix('.ig_audit.json'), audits)
    R.export_predictions(args.out, d, ids, risk)
    R.write_json(args.out.with_suffix('.metrics.json'), report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
