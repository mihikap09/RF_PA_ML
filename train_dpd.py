"""End-to-end residual DPD for RF_PA_ML. Run from the repository root.
All internal signals use train_mlp's normalized IQ units.
"""
import argparse
import copy
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import train_mlp as pa


def features(z, m):
    n = z.shape[-1]
    cols = []
    for d in range(m):
        v = z[..., m-1-d:n-d]
        cols.extend((v.real, v.imag, v.abs()))
    return torch.stack(cols, dim=-1)


class DPD(nn.Module):
    def __init__(self, m, hidden):
        super().__init__()
        self.m = m
        self.net = pa.DelayMLP(3*m, hidden)
        nn.init.zeros_(self.net.net[-1].weight)
        nn.init.zeros_(self.net.net[-1].bias)

    def forward(self, x):
        r = self.net(features(x, self.m))
        return x[..., self.m-1:] + torch.complex(r[..., 0], r[..., 1])


def forward_pa(model, u, m, gain):
    r = model(features(u, m))
    return gain*u[..., m-1:] + torch.complex(r[..., 0], r[..., 1])


def spectral_ratio(y):
    n = y.shape[-1]
    w = torch.hann_window(n, periodic=False, device=y.device)
    f = torch.fft.fftfreq(n, d=1/pa.FS, device=y.device)
    p = torch.fft.fft(y*w).abs().square()
    main = p[..., f.abs() <= pa.BW/2].sum(-1)
    lo = p[..., (f >= -1.5*pa.BW) & (f < -pa.BW/2)].sum(-1)
    hi = p[..., (f > pa.BW/2) & (f <= 1.5*pa.BW)].sum(-1)
    return (torch.maximum(lo, hi)/(main+1e-12)).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pa-depth', type=int, default=5)
    ap.add_argument('--dpd-depth', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--pa-epochs', type=int, default=200)
    ap.add_argument('--block', type=int, default=2048)
    ap.add_argument('--batch', type=int, default=4, help='Contiguous blocks per update')
    ap.add_argument('--spectral-weight', type=float, default=0.)
    ap.add_argument('--peak-weight', type=float, default=1.)
    ap.add_argument('--target-scale', type=float, default=1.)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    if min(a.pa_depth,a.dpd_depth,a.hidden,a.epochs,a.pa_epochs,a.block,a.batch,a.threads) < 1 or a.target_scale <= 0 or min(a.spectral_weight,a.peak_weight)<0:
        ap.error('Positive sizes and target scale; nonnegative loss weights required.')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    root = Path(pa.root)
    for folder in ('models','results','figures'):
        (root/folder).mkdir(exist_ok=True)
    tag = f'pa{a.pa_depth}_dpd{a.dpd_depth}_h{a.hidden}_s{a.seed}_pe{a.pa_epochs}_e{a.epochs}_sp{a.spectral_weight:g}_g{a.target_scale:g}_pk{a.peak_weight:g}_b{a.block}_lr{a.lr:g}'
    # Retrain using the original repository function, without modifying its source.
    # Original train_one runs on CPU; DPD stage can use CUDA.
    pa.EPOCHS = a.pa_epochs
    start = time.perf_counter()
    model, _, best_pa, _ = pa.train_one(a.pa_depth, a.seed)
    print(f'PA training: {time.perf_counter()-start:.1f}s; validation residual MSE {best_pa:.6g}', flush=True)
    model.to(device).eval().requires_grad_(False)
    gain = torch.tensor(complex(pa.G), dtype=torch.complex64, device=device)
    target_gain = gain*a.target_scale
    dpd = DPD(a.dpd_depth,a.hidden).to(device)
    opt = torch.optim.Adam(dpd.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=5)
    context = a.pa_depth+a.dpd_depth-2
    def normalized(x):
        return torch.as_tensor(x/pa.SX, dtype=torch.complex64, device=device)
    tr, va, te = map(normalized,(pa.x_tr,pa.x_va,pa.x_te))
    def cascade(x):
        u = dpd(x)
        y = forward_pa(model,u,a.pa_depth,gain)
        return u,y
    def loss(x):
        u,y = cascade(x)
        target = target_gain*x[...,context:]
        mse = (y-target).abs().square().mean()/(target.abs().square().mean()+1e-12)
        peak = torch.relu(u.abs()-1).square().mean()
        spec = spectral_ratio(y) if a.spectral_weight else y.real.new_zeros(())
        return mse+a.peak_weight*peak+a.spectral_weight*spec
    # Every output block has real preceding context. Never shuffle IQ samples.
    starts = torch.arange(0,tr.numel()-context-a.block+1,a.block,device=device)
    if not len(starts) or va.numel() <= context or te.numel()-context < 2048:
        raise ValueError('Insufficient data for configured memory/block sizes.')
    offsets = torch.arange(a.block+context,device=device)
    hist=[]; best=float('inf'); state=None; bad=0
    start=time.perf_counter()
    for epoch in range(a.epochs):
        dpd.train(); losses=[]
        for s in starts[torch.randperm(len(starts),device=device)].split(a.batch):
            x=tr[s[:,None]+offsets]
            opt.zero_grad(); l=loss(x); l.backward(); opt.step()
            losses.append(l.item())
        dpd.eval()
        with torch.no_grad(): v=loss(va).item()
        hist.append((epoch+1,float(np.mean(losses)),v)); sched.step(v)
        if v<best-1e-9:
            best=v; state=copy.deepcopy(dpd.state_dict()); bad=0
        else: bad+=1
        if epoch==0 or (epoch+1)%10==0:
            print(f'DPD epoch {epoch+1}: train={hist[-1][1]:.6g}, val={v:.6g}',flush=True)
        if bad>=15: break
    if state is None: raise RuntimeError('DPD training produced no finite validation loss.')
    dpd.load_state_dict(state); dpd.eval()
    print(f'DPD training: {time.perf_counter()-start:.1f}s',flush=True)
    def arr(t): return t.detach().cpu().numpy()
    with torch.no_grad():
        u,y=cascade(te)
        baseline=forward_pa(model,te,a.pa_depth,gain)[a.dpd_depth-1:]
        ideal=target_gain*te[context:]
    # Compare identical time indices, in original output units.
    target=arr(ideal)*pa.SY
    signals={'Measured PA (no DPD)':pa.y_te[context:],
             'Surrogate PA (no DPD)':arr(baseline)*pa.SY,
             'Surrogate PA (with DPD)':arr(y)*pa.SY}
    rows=[]
    for name,z in signals.items():
        acpr=float(pa.acpr_db(z))
        rows.append({'Model':name,'NMSE_to_ideal_dB':pa.nmse_db(target,z),'ACPR_dB':acpr,
                     'ACLR_dB':-acpr,'Output_mean_square':np.mean(np.abs(z)**2)})
    table=pd.DataFrame(rows)
    print(table.to_string(index=False))
    print('Surrogate PA modelling NMSE:',pa.nmse_db(signals['Measured PA (no DPD)'],signals['Surrogate PA (no DPD)']))
    print('DPD input peak (normalized):',float(u.abs().max()),'fraction above training peak:',float((u.abs()>1).float().mean()))
    print('DPD parameters:',sum(p.numel() for p in dpd.parameters()))
    print('WARNING: DPD output metrics are surrogate predictions, not measured hardware results.')
    table.to_csv(root/'results'/f'dpd_comparison_{tag}.csv',index=False)
    pd.DataFrame(hist,columns=['epoch','train_loss','val_loss']).to_csv(root/'results'/f'dpd_history_{tag}.csv',index=False)
    torch.save({'pa_state':model.cpu().state_dict(),'dpd_state':dpd.cpu().state_dict(),
                'config':vars(a),'SX':float(pa.SX),'SY':float(pa.SY),'G':complex(pa.G)},root/'models'/f'dpd_{tag}.pt')
    np.savez(root/'results'/f'dpd_waveforms_{tag}.npz', x=arr(te[context:])*pa.SX,
             u=arr(u[a.pa_depth-1:])*pa.SX, correction=(arr(u[a.pa_depth-1:])-arr(te[context:]))*pa.SX,
             target=target, measured=signals['Measured PA (no DPD)'], baseline=signals['Surrogate PA (no DPD)'], compensated=signals['Surrogate PA (with DPD)'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seg=2048; f=np.fft.fftshift(np.fft.fftfreq(seg,1/pa.FS))/1e6
    def psd(z):
        blocks=z[:len(z)//seg*seg].reshape(-1,seg)*np.hanning(seg)
        return np.mean(np.abs(np.fft.fftshift(np.fft.fft(blocks,axis=-1),axes=-1))**2,axis=0)
    ref=psd(signals['Surrogate PA (no DPD)']).max()
    for name,z in signals.items(): plt.plot(f,10*np.log10(np.maximum(psd(z)/ref,1e-15)),label=name)
    plt.xlabel('Frequency (MHz)'); plt.ylabel('PSD (dB, common reference)'); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(root/'figures'/f'dpd_psd_{tag}.png',dpi=180); plt.close()
    plt.plot([r[0] for r in hist],[r[1] for r in hist],label='Train'); plt.plot([r[0] for r in hist],[r[2] for r in hist],label='Validation')
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout(); plt.savefig(root/'figures'/f'dpd_loss_{tag}.png',dpi=180); plt.close()

if __name__=='__main__': main()
