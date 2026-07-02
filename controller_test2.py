import numpy as np, torch
import learned_continual as L
DS=['nyc','tky','gowalla_ca','brightkite']
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
for ds in DS:
    com,warm,S0=setup(ds,0)
    st=L.run_policy('s',*com,(lambda r,a,e:False),30,S0,1024,np.random.default_rng(123),warm)
    out=[f'{ds:11s} static={st["acc@10"]:.4f}']
    for eps,grow in [(0.01,1.0),(0.005,1.0),(0.01,1.1)]:
        c=L.run_controller('c',*com,ft_steps=30,lr0=1e-3,eps=eps,warm=warm,grow=grow)
        out.append(f'eps{eps}/g{grow}: {c["acc@10"]:.4f}(f{c["forget_drop"]:+.2f} lr{c["final_lr"]:.0e} rb{c["rollbacks"]})')
    print('  '.join(out))
