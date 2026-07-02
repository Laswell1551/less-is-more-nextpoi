"""Forgetting-feedback step-size controller -- the constructive deliverable. 5 seeds,
4 datasets. Shows the controller (ONE fixed eps=0.005, starting from the catastrophic
base lr 1e-3) auto-matches static at near-zero forgetting with NO per-dataset LR search,
vs naive FT (1e-3, forgets) and the manually-tuned sweet spot (3e-5, needs a grid search)."""
import json
import numpy as np, torch
import learned_continual as L
DS=['nyc','tky','gowalla_ca','brightkite']; SEEDS=[0,1,2,3,4]
def setup(ds,seed):
    df=L.load(ds); nu=int(df.user_idx.max()+1); npo=int(df.poi_idx.max()+1)
    S0=L.make_samples(df[df.block=='T0'],npo); St=L.make_samples(df[df.block!='T0'],npo)
    warm=df[df.block=='T0'].user_idx.unique(); margs=(nu,npo,128,'gru')
    torch.manual_seed(seed); np.random.seed(seed)
    b=L.NextPOI(*margs).to(L.DEVICE); L.fit(b,S0,epochs=8)
    bs={k:v.detach().cpu().clone() for k,v in b.state_dict().items()}
    rng=np.random.default_rng(seed); pj=rng.choice(len(St['tgt']),512,replace=False)
    fS=L.subset(S0,rng.choice(len(S0['tgt']),min(2000,len(S0['tgt'])),replace=False))
    rounds=np.array_split(np.arange(len(St['tgt'])),15)
    return (bs,St,rounds,pj,fS,margs),warm,S0
def ms(x): return [round(float(np.mean(x)),4),round(float(np.std(x)),4)]
out={}
for ds in DS:
    R={k:{'acc':[],'forget':[],'churn':[],'steps':[]} for k in ['static','naive_1e3','tuned_3e5','controller']}
    clr=[]; crb=[]
    for seed in SEEDS:
        com,warm,S0=setup(ds,seed)
        def add(k,r): R[k]['acc'].append(r['acc@10']); R[k]['forget'].append(r['forget_drop']); R[k]['churn'].append(r['churn']); R[k]['steps'].append(r['update_steps'])
        add('static',L.run_policy('s',*com,(lambda r,a,e:False),30,S0,1024,np.random.default_rng(123),warm))
        add('naive_1e3',L.run_policy('a',*com,(lambda r,a,e:True),30,S0,1024,np.random.default_rng(123),warm,ft_lr=1e-3))
        add('tuned_3e5',L.run_policy('a',*com,(lambda r,a,e:True),30,S0,1024,np.random.default_rng(123),warm,ft_lr=3e-5))
        c=L.run_controller('c',*com,ft_steps=30,lr0=1e-3,eps=0.005,warm=warm,grow=1.0); add('controller',c); clr.append(c['final_lr']); crb.append(c['rollbacks'])
    out[ds]={k:{'acc':ms(v['acc']),'forget':round(float(np.mean(v['forget'])),3),'churn':round(float(np.mean(v['churn'])),3),'steps':int(np.mean(v['steps']))} for k,v in R.items()}
    out[ds]['controller']['lr_range']=[float(np.min(clr)),float(np.max(clr))]; out[ds]['controller']['rollbacks']=round(float(np.mean(crb)),1)
    print(f"{ds}: static={out[ds]['static']['acc']} ctrl={out[ds]['controller']['acc']}(f{out[ds]['controller']['forget']}) naive={out[ds]['naive_1e3']['acc']}(f{out[ds]['naive_1e3']['forget']})")
json.dump(out,open(L.ROOT/'controller_eval.json','w'),indent=2)
print('-> controller_eval.json')
