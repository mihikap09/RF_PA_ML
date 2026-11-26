"""Three screening experiments for RF_PA_ML; no retraining. Run in repo root.
Requires train_dpd.py, analyse_pa_physics.py, and a trusted saved checkpoint.
Outputs hypotheses, not identified circuit changes.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch, csd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_mlp as pa
from train_dpd import DPD, forward_pa


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path)
    a=ap.parse_args(); root=Path(pa.root)
    paths=list((root/'models').glob('dpd_*.pt'))
    if not paths and a.checkpoint is None: ap.error('No checkpoint found')
    path=a.checkpoint or max(paths,key=lambda p:p.stat().st_mtime)
    ck=torch.load(path,map_location='cpu',weights_only=True)
    cfg=ck['config']; m=cfg['pa_depth']; l=cfg['dpd_depth']; g=complex(ck['G']); fs=pa.FS
    model=pa.DelayMLP(3*m,ck['pa_state']['net.0.weight'].shape[0]); model.load_state_dict(ck['pa_state']); model.eval().requires_grad_(False)
    dpd=DPD(l,cfg['hidden']); dpd.load_state_dict(ck['dpd_state']); dpd.eval().requires_grad_(False)
    torch.set_num_threads(2)
    out=root/'results'/'topology'/path.stem; out.mkdir(parents=True,exist_ok=True)
    def F(x):
        with torch.no_grad(): return forward_pa(model,torch.as_tensor(x,dtype=torch.complex64),m,g).numpy()
    def D(x):
        with torch.no_grad(): return dpd(torch.as_tensor(x,dtype=torch.complex64)).numpy()
    xv=np.asarray(pa.x_va)/ck['SX']; xt=np.asarray(pa.x_te)/ck['SX']; yt=np.asarray(pa.y_te)/ck['SY']; ctx=m+l-2
    # 1. Effective best linear response on common aligned samples. CSD convention conj(x)*y.
    x=xt[ctx:]; rows=[]; fig,ax=plt.subplots(2,1,figsize=(8,6),sharex=True)
    for label,y in [('measured',yt[ctx:]),('surrogate',F(xt)[l-1:]),('cascade',F(D(xt))),('dpd',D(xt)[m-1:])]:
        kw=dict(fs=fs,nperseg=min(512,len(x)),return_onesided=False,detrend=False)
        f,px=welch(x,**kw); _,py=welch(y,**kw); _,cross=csd(x,y,**kw)
        idx=np.argsort(f); f,px,py,cross=[v[idx] for v in (f,px,py,cross)]
        H=cross/np.maximum(px,1e-30); coh=np.abs(cross)**2/np.maximum(px*py,1e-30)
        mask=(abs(f)<=pa.BW*.45)&(px>px.max()*1e-3)&(coh>.8)
        if mask.sum()<4: print('Insufficient supported bins:',label); continue
        ff=f[mask]; phase=np.unwrap(np.angle(H[mask])); slope,intercept=np.polyfit(ff,phase,1); residual=phase-(slope*ff+intercept)
        mag=20*np.log10(np.maximum(abs(H[mask]),1e-20)); mag-=np.median(mag)
        # Use adjacent supported bins only for finite-difference group delay.
        gd=-np.diff(phase)/(2*np.pi*np.diff(ff))*1e9; adjacent=np.diff(ff)<1.5*fs/kw['nperseg']; gd=gd[adjacent]
        rows.append(dict(signal=label,supported_bins=int(mask.sum()),gain_ripple_db=np.ptp(mag),bulk_delay_ns=-slope/(2*np.pi)*1e9,residual_phase_rms_deg=np.sqrt(np.mean(residual**2))*180/np.pi,group_delay_p90_p10_ns=np.ptp(np.percentile(gd,[10,90])) if len(gd) else np.nan,median_coherence=np.median(coh[mask])))
        ax[0].plot(ff/1e6,mag,label=label); ax[1].plot(ff/1e6,residual*180/np.pi,label=label)
        pd.DataFrame(dict(f_mhz=ff/1e6,magnitude_relative_db=mag,residual_phase_deg=residual*180/np.pi,coherence=coh[mask])).to_csv(out/f'response_{label}.csv',index=False)
    ax[0].set_ylabel('Relative gain (dB)'); ax[1].set_ylabel('Residual phase (deg)'); ax[1].set_xlabel('Baseband MHz')
    for q in ax:q.grid(alpha=.3);q.legend()
    fig.tight_layout();fig.savefig(out/'frequency_response.png',dpi=160);plt.close(fig)
    df=pd.DataFrame(rows);df.to_csv(out/'frequency_summary.csv',index=False);print('\n1. Effective linear response\n',df.to_string(index=False))
    # 2. Two-tone coherent sweep with common-phase ensemble. Surrogate extrapolation.
    N=16384;n=np.arange(N+m-1);rows=[]
    for peak in [.25,.5,.75]:
        for spacing in [1,2,5,10,20,40,80,100]:
            k=max(1,round(spacing*1e6*N/(2*fs))); powers=[]
            for ph in np.arange(4)*np.pi/2:
                z=peak*np.cos(2*np.pi*k*n/N)*np.exp(1j*ph);Y=np.fft.fft(F(z))/N
                powers.append([abs(Y[j%N])**2 for j in [-k,k,-3*k,3*k]])
            p=np.asarray(powers); db=10*np.log10(np.maximum(p[:,2:],1e-30)/np.maximum(p[:,:2],1e-30));lo,hi=10*np.log10(np.maximum(p[:,2:].mean(0),1e-30)/np.maximum(p[:,:2].mean(0),1e-30))
            rows.append(dict(peak=peak,spacing_mhz=2*k*fs/N/1e6,lower_imd3_dbc=lo,upper_imd3_dbc=hi,asymmetry_db=hi-lo,phase_spread_db=np.max(np.ptp(db,axis=0))))
    df=pd.DataFrame(rows);df.to_csv(out/'two_tone_screen.csv',index=False);print('\n2. Two-tone screen (surrogate only)\n',df.to_string(index=False))
    for peak,d in df.groupby('peak'):
        plt.semilogx(d.spacing_mhz,d.lower_imd3_dbc,'o-',label=f'lower {peak}');plt.semilogx(d.spacing_mhz,d.upper_imd3_dbc,'s--',label=f'upper {peak}')
    plt.xlabel('Spacing (MHz)');plt.ylabel('IMD3 (dBc)');plt.legend(fontsize=8);plt.grid(alpha=.3);plt.tight_layout();plt.savefig(out/'two_tone_screen.png',dpi=160);plt.close()
    # 3. Fit residual polynomial on validation, modify static order groups on test.
    # Include memory terms in joint fit; preserve unexplained residual. Coefficients correlated.
    orders=[1,3,5,7]; labels=[(p,j) for p in orders for j in range(m)]
    def basis(z):
        w=np.lib.stride_tricks.sliding_window_view(z,m)[:,::-1];return np.column_stack([w[:,j]*abs(w[:,j])**(p-1) for p,j in labels])
    A=basis(xv); B=basis(xt); rv=F(xv)-g*xv[m-1:]; y=F(xt); rt=y-g*xt[m-1:]; scale=np.maximum(np.sqrt(np.mean(abs(A)**2,axis=0)),1e-12); C=A/scale
    coef=np.linalg.lstsq(C,rv,rcond=1e-8)[0]/scale
    groups={'static_cubic':[(3,0)],'static_high_order':[(5,0),(7,0)],'all_static_nonlinear':[(p,0) for p in orders if p>1],'nonlinear_memory':[(p,j) for p,j in labels if p>1 and j>0]}
    rows=[];base=-pa.acpr_db(y);ref=g*xt[m-1:]
    for name,terms in groups.items():
        idx=[labels.index(t) for t in terms];v=B[:,idx]@coef[idx]
        for reduction in [0,.2,.5,1.]:
            z=y-reduction*v;ac=-pa.acpr_db(z);rows.append(dict(component=name,reduction=reduction,aclr_db=ac,aclr_change_db=ac-base,nmse_db=pa.nmse_db(ref,z),power_change_db=10*np.log10(np.mean(abs(z)**2)/np.mean(abs(y)**2))))
    df=pd.DataFrame(rows);df.to_csv(out/'static_interventions.csv',index=False);print('\n3. Mathematical distortion interventions\n',df.to_string(index=False));print('Decomposition NMSE:',pa.nmse_db(rt,B@coef),'condition:',np.linalg.cond(C))
    for name,d in df.groupby('component'):plt.plot(d.reduction,d.aclr_change_db,'o-',label=name)
    plt.xlabel('Fraction removed');plt.ylabel('Predicted ACLR change (dB)');plt.legend(fontsize=8);plt.grid(alpha=.3);plt.tight_layout();plt.savefig(out/'static_interventions.png',dpi=160);plt.close()
    print('\nSaved:',out,'\nHypotheses only: effective linear response includes measurement chain; two-tone extrapolation is unvalidated; polynomial orders are correlated and not physical circuit parameters. No PAE or topology benefit is established.')

if __name__=='__main__':main()
