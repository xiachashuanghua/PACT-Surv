"""One five-cancer protocol: train-only CV, target-selected source, frozen/full/off FT."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import pickle
import sys

for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='1'
import numpy as np
import torch
from sklearn.model_selection import RepeatedStratifiedKFold
import io_utils as R
import pipeline as P
import model as M
import linear as L

ROOT=Path(__file__).resolve().parent
OUT=Path('results').resolve()
SOURCES=Path('sources').resolve()
CANCERS=['luad','lgg','kirc','brca','lihc']
WDS=[.001,.01,.1]
STEPS=[25,50,100,200,400,800]
CONFIGS=[(mode,lr,beta,step) for mode,lrs in [('frozen',[.001]),('full',[.00001,.0001])]
         for lr in lrs for beta in [.01,1.] for step in STEPS]
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def protocol():
    return dict(cancers=CANCERS,folds=5,repeats=2,score='mean training-CV C-index',
        validation='report only',source_dropout=.5,source_lr=1e-5,source_wd=WDS,source_seeds=list(range(5)),
        source_exclusion='target cancer only',linear_candidates=972,ft_candidates=CONFIGS,
        source_selection='3 five-network ensembles compared by frozen FT lr=.001 beta=1 steps=25/50/100/200/400/800 on target training CV; each ensemble uses its best mean-CV step',
        transfer_selection='off plus 36 frozen/full candidates on selected ensemble, same CV folds',
        preprocessing='per fitting-fold RNA/clinical imputation and standardization for all five cancers',
        neural_precision='float32, TF32 disabled',cox_calculation='logcumsumexp stable Breslow likelihood',dropout_during_target_ft=0,
        maximum_target_steps=800,gate_lr=.001,
        limitation='staged tuning reuses CV folds; no claim of fully nested selection')

def source_folder(c,wd): return SOURCES/c/f'p0.5_wd{wd}_lr1e-05'

def complete_sources(c):
    import pretrain19 as B
    import pretrain19b as A
    missing=[]
    for wd in WDS:
        for seed in range(5):
            p=source_folder(c,wd)/f's{seed}'
            present=[p.with_suffix('.npz').exists(),p.with_suffix('.json').exists()]
            if any(present) and not all(present):
                raise RuntimeError(f'Incomplete source artifact: {p}; inspect before resuming')
            if not all(present): missing.append((wd,seed))
    if not missing: return
    with np.load(SOURCES/'pancan_all.npz') as f:
        keep=f['type']!=c.upper()
        X,t,e,g,patients=[f[k][keep] for k in ['X','time','event','type','patients']]
        genes=f['genes'].copy()
    target=P.load_data(c)
    np.testing.assert_array_equal(genes,target['broad_genes'])
    assert not set(patients)&set(target['patients'])
    assert c.upper() not in set(g) and set(C.upper() for C in CANCERS if C!=c)<=set(g)
    assert np.isfinite(X).all() and np.isfinite(t).all() and (t>0).all()
    assert np.isin(e,[0,1]).all() and len(set(patients))==len(patients)
    X=X.astype(np.float32); tr,va=B.split(t,e,g)
    parent=SOURCES/c; parent.mkdir(parents=True,exist_ok=True)
    split_path=parent/'split.npz'
    if split_path.exists():
        with np.load(split_path) as saved:
            for name,value in [('train',tr),('val',va),('patients',patients),('genes',genes)]:
                np.testing.assert_array_equal(saved[name],value)
    else:
        np.savez(split_path,train=tr,val=va,patients=patients,genes=genes)
    B.P_HID,B.EVERY=.5,1
    for wd,s in missing:
        B.WD=wd
        W,info,hist=A.fit('real',s,1e-5,X,t,e,g,tr,va,steps=5000,patience=500)
        folder=source_folder(c,wd); folder.mkdir(parents=True,exist_ok=True)
        np.savez(folder/f's{s}.npz',**W)
        info.update(dropout=.5,weight_decay=wd,target_exclusion=c,source_sha256=R.sha(SOURCES/'pancan_all.npz'))
        np.savetxt(folder/f's{s}.csv',hist,delimiter=',',header='step,train_loss,val_loss',comments='')
        R.write_json(folder/f's{s}.json',info)
        print('SOURCE_COMPLETED',c,wd,s,flush=True)


def sources(c):
    result={}
    for wd in WDS:
        nets=[]
        for s in range(5):
            with np.load(source_folder(c,wd)/f's{s}.npz') as f: nets.append({k:f[k].copy() for k in f.files})
        result[wd]=nets
    return result

def signature(c):
    return dict(base=R.signature(),data=R.sha(P.DATA_DIR/f'{c}.npz'),protocol=protocol(),
        source={f'{wd}/{s}':R.sha(source_folder(c,wd)/f's{s}.npz') for wd in WDS for s in range(5)})

def filled(d,fit):
    # Fit only RNA imputation; linear_blocks never consumes other modalities here.
    raw=d['x'][:,0]
    mean=np.nan_to_num(np.nanmean(raw[fit],axis=0),nan=0.)
    x=np.where(np.isnan(raw),mean,raw)[:,None,:]
    return x,mean

def network_input(d,fit,held):
    raw=d['full_rna'][:,d['gene_idx']].astype(float)
    impute=np.nan_to_num(np.nanmean(raw[fit],axis=0),nan=0.)
    full=np.where(np.isnan(raw),impute,raw)
    mu,sd=P.scale_fit(full[fit],'unified')
    return [np.clip((full[i]-mu)/sd,-5,5).astype(np.float32) for i in [fit,*held]],dict(impute=impute,mu=mu,sd=sd)

class Network(torch.nn.Module):
    def __init__(self,w):
        super().__init__()
        self.l1=torch.nn.Linear(w['w1'].shape[1],w['w1'].shape[0])
        self.l2=torch.nn.Linear(w['w2'].shape[1],w['w2'].shape[0])
        self.l3=torch.nn.Linear(w['w3'].shape[1],1,bias=False)
        with torch.no_grad():
            for name in ['1','2','3']:
                layer=getattr(self,'l'+name); layer.weight.copy_(torch.as_tensor(w['w'+name]))
                if layer.bias is not None: layer.bias.copy_(torch.as_tensor(w['b'+name]))
        self.mu=float(w['risk_mean']); self.sd=float(w['risk_sd'])
    def forward(self,z):
        h=torch.nn.functional.gelu(self.l1(z))
        return (self.l3(torch.nn.functional.gelu(self.l2(h))).squeeze(-1)-self.mu)/self.sd

class StableCox(M.CoxT):
    def __call__(self,r):
        r=r[self.o]
        log_risk=torch.logcumsumexp(r,dim=0)[self.last]
        return (self.e*(log_risk-r)).sum()/self.ne


def trajectories(blocks,off,held,t,e,nets,configs,save=False):
    z=[torch.as_tensor(x,device=DEV) for x in blocks]
    loss_fn=StableCox(t,e)
    loss_fn.o=loss_fn.o.to(DEV); loss_fn.last=loss_fn.last.to(DEV); loss_fn.e=loss_fn.e.to(DEV)
    offset=torch.as_tensor(off,dtype=torch.float32,device=DEV)
    result={}; states={}
    for mode,lr,beta in dict.fromkeys(cfg[:3] for cfg in configs):
        heads=[Network(w).to(DEV) for w in nets]
        gate=torch.zeros(1,device=DEV,requires_grad=True)
        if mode=='frozen':
            for head in heads:
                for p in head.l1.parameters(): p.requires_grad_(False)
            with torch.no_grad(): inputs=[[torch.nn.functional.gelu(h.l1(v)) for h in heads] for v in z]
        params=[p for h in heads for p in h.parameters() if p.requires_grad]
        prior=[p.detach().clone() for p in params]
        opt=torch.optim.Adam([{'params':params,'lr':lr},{'params':[gate],'lr':.001}])
        def output(index):
            if mode=='full': outputs=[h(z[index]) for h in heads]
            else: outputs=[(h.l3(torch.nn.functional.gelu(h.l2(v))).squeeze(-1)-h.mu)/h.sd for h,v in zip(heads,inputs[index])]
            return gate*torch.stack(outputs).mean(0)
        steps=[cfg[3] for cfg in configs if cfg[:3]==(mode,lr,beta)]
        for step in range(1,max(steps)+1):
            opt.zero_grad()
            loss=loss_fn(offset+output(0))+beta*sum(((p-q)**2).sum() for p,q in zip(params,prior))
            assert torch.isfinite(loss)
            loss.backward(); opt.step()
            if step in steps:
                key=(mode,lr,beta,step)
                with torch.no_grad(): result[key]=[output(i).cpu().numpy().copy()+base for i,base in enumerate([off,*held])]
                if save: states[key]=dict(gate=float(gate.detach().cpu()),weights=[{k:v.detach().cpu().numpy().copy() for k,v in h.state_dict().items()} for h in heads])
        del heads,opt,params,prior
        if DEV.type=='cuda': torch.cuda.empty_cache()
    return result,states

def score(r,d,ids): return L.ci(r,d['time'][ids],d['event'][ids])

def predict(d,ids,state,nets):
    raw=d['x'][:,0]; x=np.where(np.isnan(raw),state['rna_impute'],raw)
    s=state['linear']; clinic=(np.where(np.isnan(d['clinical'][ids]),s['clinical_median'],d['clinical'][ids])-s['clinical_mean'])/s['clinical_sd']
    rna=(x[ids]-s['rna_mean'])/s['rna_sd']
    risk=np.column_stack([clinic,rna[:,s['idx'][:s['k']]]])@s['coef']
    if state['ft'] is not None:
        f=state['network_prep']; raw=d['full_rna'][ids][:,d['gene_idx']]
        z=np.clip((np.where(np.isnan(raw),f['impute'],raw)-f['mu'])/f['sd'],-5,5).astype(np.float32)
        heads=[Network(w).to(DEV) for w in nets]
        for h,w in zip(heads,state['ft']['weights']): h.load_state_dict({k:torch.as_tensor(v,device=DEV) for k,v in w.items()})
        with torch.no_grad(): correction=torch.stack([h(torch.as_tensor(z,device=DEV)) for h in heads]).mean(0).cpu().numpy()
        risk=risk+state['ft']['gate']*correction
    assert np.isfinite(risk).all()
    return risk

def train(c):
    sig=signature(c); allnets=sources(c)
    d=P.load_data(c); raw_d=d.copy(); d['cancer']='unified'
    for split in args.splits:
        folder=OUT/c/f'split_{split:02}'; folder.mkdir(parents=True,exist_ok=True)
        if (folder/'selection.json').exists():
            assert json.loads((folder/'selection.json').read_text())['signature']==json.loads(json.dumps(sig))
            continue
        _,ids=P.split_data(raw_d,split); tr,va=ids['train'],ids['val']
        folds=[(tr[a],tr[b]) for a,b in RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=12000+split).split(tr,d['event'][tr])]
        records=[]
        cache=folder/'linear_cv.pkl'
        if cache.exists():
            saved=pickle.loads(cache.read_bytes()); assert saved['signature']==sig
            scores=saved['scores']; records=saved['records']
        else:
            scores={}
            for f,(a,b) in enumerate(folds):
                x,_=filled(d,a)
                for cfg,_,risk,held,_ in P.candidates(d,x,a,[b]):
                    value=score(held[0],d,b); scores.setdefault(cfg,[]).append(value)
                    records.append(dict(stage='linear',fold=f,config=json.dumps(cfg),c_index=value))
                print('LINEAR_CV',c,split,f+1,flush=True)
            assert len(scores)==972 and all(len(v)==10 for v in scores.values())
            cache.write_bytes(pickle.dumps(dict(signature=sig,scores=scores,records=records)))
        if args.linear_only: continue
        x,impute=filled(d,tr)
        for cfg in sorted(scores,key=lambda k:-np.mean(scores[k])):
            linear,off,held=P.fit_linear(d,x,tr,[va],cfg)
            if np.ptp(off)>1e-12: break
        else: raise RuntimeError('constant linear model')
        source_step_scores={(wd,step):[] for wd in WDS for step in STEPS}; ft_scores={None:[]}
        probes=[('frozen',.001,1.,step) for step in STEPS]
        fold_cache=[]
        for f,(a,b) in enumerate(folds):
            fx,_=filled(d,a); _,base,h=P.fit_linear(d,fx,a,[b],cfg)
            blocks,_=network_input(d,a,[b]); fold_cache.append((a,b,base,h,blocks))
            for wd in WDS:
                result,_=trajectories(blocks,base,h,d['time'][a],d['event'][a],allnets[wd],probes)
                for probe in probes:
                    value=score(result[probe][1],d,b); source_step_scores[wd,probe[3]].append(value)
                    records.append(dict(stage='source',fold=f,config=json.dumps([wd,probe[3]]),c_index=value))
            print('SOURCE_CV',c,split,f+1,flush=True)
        source_scores={wd:max((source_step_scores[wd,step] for step in STEPS),key=np.mean) for wd in WDS}
        wd=max(WDS,key=lambda k:np.mean(source_scores[k])); nets=allnets[wd]
        for cfg_ft in CONFIGS: ft_scores[cfg_ft]=[]
        for f,(a,b,base,h,blocks) in enumerate(fold_cache):
            result,_=trajectories(blocks,base,h,d['time'][a],d['event'][a],nets,CONFIGS)
            for candidate in ft_scores:
                value=score(h[0] if candidate is None else result[candidate][1],d,b)
                ft_scores[candidate].append(value)
                records.append(dict(stage='transfer',fold=f,config=json.dumps(candidate),c_index=value))
            print('TRANSFER_CV',c,split,f+1,flush=True)
        ft=max(ft_scores,key=lambda k:np.mean(ft_scores[k]))
        blocks,prep=network_input(d,tr,[va]); chosen=None
        if ft is not None:
            result,states=trajectories(blocks,off,held,d['time'][tr],d['event'][tr],nets,[ft],True); chosen=states[ft]
        state=dict(linear=linear,ft=chosen,network_prep=prep,rna_impute=impute,ids=ids,source_wd=wd,signature=sig)
        (folder/'model.pkl').write_bytes(pickle.dumps(state,protocol=4))
        for part in ['train','val']: R.export_predictions(folder/f'{part}_predictions.csv',d,ids[part],predict(d,ids[part],state,nets))
        R.csv_rows(folder/'cv_scores.csv',records)
        R.write_json(folder/'selection.json',dict(cancer=c,split=split,linear_config=cfg,source_wd=wd,transfer=ft,
            source_cv={str(k):float(np.mean(v)) for k,v in source_scores.items()},source_step_cv={json.dumps(k):float(np.mean(v)) for k,v in source_step_scores.items()},transfer_cv={json.dumps(k):float(np.mean(v)) for k,v in ft_scores.items()},
            signature=sig,model_sha256=R.sha(folder/'model.pkl')))
        print('MODEL_FROZEN',c,split,wd,ft,flush=True)

def evaluate():
    expected=[OUT/c/f'split_{s:02}'/name for c in CANCERS for s in range(1,11) for name in ['model.pkl','selection.json']]
    assert all(p.exists() for p in expected),'finish all 50 models before test evaluation'
    frozen={str(p.relative_to(OUT)):R.sha(p) for p in expected}
    manifest=OUT/'evaluation_manifest.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==frozen
    else: R.write_json(manifest,frozen)
    import metrics as metric
    summaries=[]
    for c in CANCERS:
        sig=signature(c); nets=sources(c); d=P.load_data(c); rows=[]
        for s in range(1,11):
            folder=OUT/c/f'split_{s:02}'; selection=json.loads((folder/'selection.json').read_text())
            state=pickle.loads((folder/'model.pkl').read_bytes()); assert state['signature']==sig
            assert selection['model_sha256']==R.sha(folder/'model.pkl')
            ids=state['ids']; _,expected_ids=P.split_data(d,s)
            for part in ['train','val','test']:
                np.testing.assert_array_equal(ids[part],expected_ids[part])
                risk=predict(d,ids[part],state,nets[state['source_wd']])
                dest=folder/f'{part}_predictions.csv'
                if dest.exists():
                    with dest.open() as f: old=list(csv.DictReader(f))
                    assert [x['patient'] for x in old]==list(d['patients'][ids[part]])
                    np.testing.assert_allclose([float(x['risk']) for x in old],risk,atol=1e-6,rtol=1e-6)
                else: assert part=='test'; R.export_predictions(dest,d,ids[part],risk)
                if part=='test': rows.append(dict(cancer=c,split=s,c_index=score(risk,d,ids[part]),pooled_auc=metric.pooled_auc(risk,d['time'][ids[part]],d['event'][ids[part]])))
        R.csv_rows(OUT/c/'test_results.csv',rows)
        summary=dict(cancer=c,splits=10,c_mean=float(np.mean([x['c_index'] for x in rows])),c_sd=float(np.std([x['c_index'] for x in rows],ddof=1)),
            auc_mean=metric.summary([x['pooled_auc'] for x in rows])['mean'],auc_sd=metric.summary([x['pooled_auc'] for x in rows])['sample_sd'],auc_defined_splits=metric.summary([x['pooled_auc'] for x in rows])['n_defined'])
        R.write_json(OUT/c/'summary.json',summary); R.write_json(OUT/c/'verification.json',dict(status='PASS',splits=10))
        summaries.append(summary)
    R.write_json(OUT/'summary.json',summaries); R.csv_rows(OUT/'summary.csv',summaries)
    print('EVALUATION_COMPLETE',json.dumps(summaries),flush=True)

def self_test():
    assert STEPS==[25,50,100,200,400,800] and len(CONFIGS)==36
    t0=np.array([5.,4.,4.,1.]); e0=np.array([1.,0.,1.,1.])
    ordinary=torch.tensor([.2,-.3,.7,.1],dtype=torch.float64,requires_grad=True)
    old=M.CoxT(t0,e0)(ordinary); stable=StableCox(t0,e0)(ordinary)
    torch.testing.assert_close(old,stable)
    torch.testing.assert_close(torch.autograd.grad(old,ordinary)[0],torch.autograd.grad(stable,ordinary)[0])
    extreme=torch.tensor([-1000.,0.,1000.,2000.],requires_grad=True)
    loss=StableCox(t0,e0)(extreme); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(extreme.grad).all()
    rng=np.random.default_rng(3); n=12
    w=dict(w1=rng.normal(size=(5,8)).astype(np.float32)*.1,b1=np.zeros(5),w2=rng.normal(size=(3,5)).astype(np.float32)*.1,
           b2=np.zeros(3),w3=rng.normal(size=(1,3)).astype(np.float32)*.1,risk_mean=np.array(0.),risk_sd=np.array(1.))
    z=rng.normal(size=(n,8)).astype(np.float32); t=np.arange(1,n+1,dtype=float); e=np.ones(n); off=rng.normal(size=n)
    for mode in ['frozen','full']:
        cfg=(mode,.001,.01,3)
        result,states=trajectories([z,z],off,[off],t,e,[w]*5,[cfg],True)
        h=Network(w).to(DEV); h.load_state_dict({k:torch.as_tensor(v,device=DEV) for k,v in states[cfg]['weights'][0].items()})
        np.testing.assert_array_equal(states[cfg]['weights'][0]['l1.weight'],w['w1']) if mode=='frozen' else None
        if mode=='full': assert not np.array_equal(states[cfg]['weights'][0]['l1.weight'],w['w1'])
        with torch.no_grad(): expected=off+states[cfg]['gate']*h(torch.as_tensor(z,device=DEV)).cpu().numpy()
        np.testing.assert_allclose(expected,result[cfg][0],atol=1e-6,rtol=1e-6)
    raw=np.ones((4,1,2)); raw[0,0,0]=np.nan; raw[3,0,1]=99
    x,mean=filled({'x':raw},np.array([0,1])); assert mean[1]==1
    print('SELF_TEST_PASS frozen/full gradient, model reload, fitting-only imputation',flush=True)

def main():
    global args, OUT, SOURCES, DEV
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    for name in ['train','evaluate','self-test','prepare-sources']:
        mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--cancer',choices=CANCERS)
    parser.add_argument('--device',choices=['cpu','cuda'],default=None)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--source-dir',type=Path,default=Path('sources'))
    parser.add_argument('--out-dir',type=Path,default=Path('results'))
    parser.add_argument('--linear-only',action='store_true')
    parser.add_argument('--splits',type=int,nargs='+',default=list(range(1,11)),choices=range(1,11))
    args=parser.parse_args()
    if not args.self_test and not args.evaluate and not args.cancer:
        parser.error('--cancer is required for training or pretraining')
    if len(set(args.splits))!=len(args.splits): parser.error('duplicate splits')
    P.DATA_DIR=args.data_dir.resolve(); SOURCES=args.source_dir.resolve(); OUT=args.out_dir.resolve()
    # Set CUDA_VISIBLE_DEVICES in the shell before launch if selecting a GPU.
    DEV=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    import pretrain19 as B
    B.DEV=DEV
    if args.self_test: self_test(); return
    if args.prepare_sources: complete_sources(args.cancer); return
    OUT.mkdir(parents=True,exist_ok=True)
    if args.evaluate: evaluate(); return
    stamp=OUT/'protocol.json'
    if stamp.exists(): assert json.loads(stamp.read_text())==json.loads(json.dumps(protocol()))
    else: R.write_json(stamp,protocol())
    train(args.cancer)


if __name__=='__main__': main()
