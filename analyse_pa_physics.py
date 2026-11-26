"""Exploratory physics diagnostics for RF_PA_ML. No model retraining.
Run in repo root: python analyse_pa_physics.py --checkpoint models/dpd_....pt
All model calculations use normalized IQ. Circuit hypotheses are not causal diagnoses.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_mlp as pa
from train_dpd import DPD, forward_pa, features


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path)
    args=ap.parse_args()
    root=Path(pa.root)
    paths=list((root/'models').glob('dpd_*.pt'))
    if args.checkpoint is None and not paths: ap.error('No DPD checkpoints found')
    path=args.checkpoint or max(paths,key=lambda p:p.stat().st_mtime)
    # Only load trusted checkpoints produced by your own training script.
    ck=torch.load(path,map_location='cpu',weights_only=True)
    cfg=ck['config']; m=cfg['pa_depth']; l=cfg['dpd_depth']; sx=ck['SX']; sy=ck['SY']; g=complex(ck['G'])
    hidden=ck['pa_state']['net.0.weight'].shape[0]
    model=pa.DelayMLP(3*m,hidden); model.load_state_dict(ck['pa_state']); model.eval().requires_grad_(False)
    dpd=DPD(l,cfg['hidden']); dpd.load_state_dict(ck['dpd_state']); dpd.eval().requires_grad_(False)
    torch.set_num_threads(2)
    out=root/'results'/'physics'/path.stem; out.mkdir(parents=True,exist_ok=True)
    print('Checkpoint:',path,'\nOutput:',out)
    def tensor(z): return torch.as_tensor(z,dtype=torch.complex64)
    def arr(z): return z.detach().cpu().numpy()
    def F(z):
        with torch.no_grad(): return arr(forward_pa(model,tensor(z),m,g))
    def D(z):
        with torch.no_grad(): return arr(dpd(tensor(z)))
    def aclr(z): return -float(pa.acpr_db(z))
    def nmse(ref,z): return float(pa.nmse_db(ref,z))
    def save(name): plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/(name+'.png'),dpi=160); plt.close()
    xv=np.asarray(pa.x_va)/sx; xt=np.asarray(pa.x_te)/sx; yt=np.asarray(pa.y_te)/sy
    ctx=m+l-2; uv=D(xv); ut=D(xt); base=F(xt)[l-1:]; comp=F(ut); measured=yt[ctx:]; ideal=g*cfg.get('target_scale',1)*xt[ctx:]
    print('Measured / surrogate / compensated ACLR:',*[round(aclr(z),3) for z in (measured,base,comp)])
    # 1. Conditional complex gain, using LS within amplitude bins to avoid division near zero.
    rows=[]; edges=np.linspace(0,max(abs(xt[ctx:]).max(),.01),21)
    for label,y in [('Measured PA',measured),('Surrogate PA',base),('DPD cascade',comp)]:
        x=xt[ctx:]
        for a,b in zip(edges[:-1],edges[1:]):
            mask=(abs(x)>=a)&(abs(x)<b)
            if mask.sum()<20: continue
            h=np.vdot(x[mask],y[mask])/np.vdot(x[mask],x[mask]); h/=g
            rows.append((label,(a+b)/2,int(mask.sum()),20*np.log10(max(abs(h),1e-15)),np.angle(h,deg=True)))
    df=pd.DataFrame(rows,columns=['signal','amplitude','count','relative_gain_db','relative_phase_deg']); df.to_csv(out/'am_am_pm.csv',index=False)
    for col in ['relative_gain_db','relative_phase_deg']:
        for label,d in df.groupby('signal'): plt.plot(d.amplitude,d[col],'.-',label=label)
        plt.xlabel('Normalized original input amplitude'); plt.ylabel(col); plt.legend(); save(col)
    # 2. Total derivative wrt actual I/Q (including magnitude dependence).
    z=tensor(xt); windows=z.unfold(0,m,1)[::max(1,(len(z)-m+1)//2048)].flip(-1).contiguous(); q=torch.view_as_real(windows).clone().requires_grad_(True)
    zz=torch.view_as_complex(q); feat=torch.stack((zz.real,zz.imag,zz.abs()),dim=-1).flatten(-2)
    r=model(feat); linear=g*zz[:,0]; full=r+torch.stack((linear.real,linear.imag),dim=-1)
    sens=[]
    for name,o in [('residual',r),('full',full)]:
        grads=[torch.autograd.grad(o[:,j].sum(),q,retain_graph=True)[0] for j in range(2)]
        v=torch.stack(grads).square().sum(dim=(0,3)).mean(dim=0).detach().numpy()
        for j,s in enumerate(v): sens.append((name,j,j/pa.FS*1e9,float(s)))
    df=pd.DataFrame(sens,columns=['model','delay','delay_ns','mean_jacobian_frobenius_squared']); df.to_csv(out/'memory_sensitivity.csv',index=False)
    for label,d in df.groupby('model'): plt.plot(d.delay_ns,d.mean_jacobian_frobenius_squared,'.-',label=label)
    plt.xlabel('Delay (ns)'); plt.ylabel('Mean squared Jacobian norm'); plt.legend(); save('memory_sensitivity')
    # 3. Fit interpretable MP basis on validation only; evaluate/ablate on held-out test.
    orders=[1,3,5,7]; groups=['linear_current','nonlinear_current','linear_memory','nonlinear_memory']
    def basis(x,depth):
        w=np.lib.stride_tricks.sliding_window_view(x,depth)[:,::-1]; cols=[]; labels=[]
        for p in orders:
            for j in range(depth):
                cols.append(w[:,j]*abs(w[:,j])**(p-1)); labels.append((p,j))
        return np.column_stack(cols),labels
    def group(p,j): return ('linear' if p==1 else 'nonlinear')+('_current' if j==0 else '_memory')
    def fit(xv,xt,rv,rt,depth,tag):
        A,labels=basis(xv,depth); B,_=basis(xt,depth)
        scale=np.maximum(np.sqrt(np.mean(abs(A)**2,axis=0)),1e-12)
        coef=np.linalg.lstsq(A/scale,rv,rcond=1e-8)[0]/scale
        prediction=B@coef; parts={k:np.zeros(len(B),complex) for k in groups}; terms=[]
        for i,(p,j) in enumerate(labels):
            v=B[:,i]*coef[i]; parts[group(p,j)]+=v; terms.append((p,j,coef[i].real,coef[i].imag,np.mean(abs(v)**2)))
        pd.DataFrame(terms,columns=['order','delay','coef_real','coef_imag','term_mean_square']).to_csv(out/(tag+'_coefficients.csv'),index=False)
        print(tag,'held-out decomposition NMSE:',nmse(rt,prediction),'normalized basis condition:',np.linalg.cond(A/scale))
        return parts,prediction
    dc_v=uv-xv[l-1:]; dc_t=ut-xt[l-1:]
    parts,pred=fit(xv,xt,dc_v,dc_t,l,'dpd')
    rows=[('full DPD',aclr(comp),nmse(ideal,comp),np.mean(abs(comp)**2))]
    for k,v in parts.items():
        y=F(ut-v); rows.append(('remove '+k,aclr(y),nmse(ideal,y),np.mean(abs(y)**2)))
    df=pd.DataFrame(rows,columns=['intervention','aclr_db','nmse_to_ideal_db','output_mean_square']); df['aclr_change_db']=df.aclr_db-df.aclr_db.iloc[0]; df.to_csv(out/'dpd_ablation.csv',index=False); print('\nDPD ablation\n',df.to_string(index=False))
    # 4. Virtual PA interventions on forward residual, retaining unexplained residual.
    rv=F(xv)-g*xv[m-1:]; rt=F(xt)-g*xt[m-1:]; pp,_=fit(xv,xt,rv,rt,m,'pa_residual')
    y0=F(xt); rows=[]
    for k,v in pp.items():
        for remaining in [1.,.8,.5,0.]:
            y=y0-(1-remaining)*v; rows.append((k,remaining,aclr(y),nmse(g*xt[m-1:],y),np.mean(abs(y)**2)))
    df=pd.DataFrame(rows,columns=['component','remaining_fraction','aclr_db','nmse_to_linear_db','output_mean_square']); df['aclr_change_db']=df.aclr_db-aclr(y0); df.to_csv(out/'virtual_pa_interventions.csv',index=False)
    for k,d in df.groupby('component'): plt.plot(d.remaining_fraction,d.aclr_change_db,'.-',label=k)
    plt.xlabel('Remaining component fraction'); plt.ylabel('Predicted ACLR change (dB)'); plt.legend(); save('virtual_pa_interventions')
    # 5. Coherent two-tone sweep, exact FFT bins, no window needed for periodic steady state.
    N=16384; n=np.arange(N+m-1); rows=[]
    for peak in [.25,.5,.75]:
        for spacing in [1,2,5,10,20,40,80,100]:
            k=max(1,round(spacing*1e6*N/(2*pa.FS))); z=(peak/2)*(np.exp(-2j*np.pi*k*n/N)+np.exp(2j*np.pi*k*n/N)); y=F(z); Y=np.fft.fft(y)/N
            def power(j): return abs(Y[j%N])**2
            def db(a,b): return 10*np.log10(max(a,1e-30)/max(b,1e-30))
            rows.append((peak,2*k*pa.FS/N/1e6,db(power(-3*k),power(-k)),db(power(3*k),power(k)),db(power(-5*k),power(-k)),db(power(5*k),power(k))))
    df=pd.DataFrame(rows,columns=['input_peak','spacing_mhz','imd3_lower_dbc','imd3_upper_dbc','imd5_lower_dbc','imd5_upper_dbc']); df.to_csv(out/'two_tone.csv',index=False)
    for peak,d in df.groupby('input_peak'):
        plt.plot(d.spacing_mhz,d.imd3_lower_dbc,'.-',label=f'lower peak {peak}'); plt.plot(d.spacing_mhz,d.imd3_upper_dbc,'.--',label=f'upper peak {peak}')
    plt.xscale('log'); plt.xlabel('Tone spacing (MHz)'); plt.ylabel('IMD3 (dBc)'); plt.legend(fontsize=8); save('two_tone_imd3')
    print('\nInterpretation: use ablation and intervention tables to prioritize static versus dynamic distortion. Two-tone results are unvalidated surrogate extrapolations. Polynomial components are correlated, not unique physical mechanisms. No circuit parameter values or PAE can be inferred from these data alone.')

if __name__=='__main__': main()
