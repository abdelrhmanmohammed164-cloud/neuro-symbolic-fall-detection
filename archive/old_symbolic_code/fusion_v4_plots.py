import json,sys,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'/home/claude/neuro-symbolic-fall-detection/fall_detection_models/code/stgcn_urfd')
import fusion_v4_experiments as X, symbolic_v4 as S4, symbolic_v3 as S3
import os; os.environ['PROJECT_ROOT']='/tmp/proot'
X.ROOT='/tmp/proot'; X.NPY=X.ROOT+'/processed_data/cfg_d31bfed741ab/urfd_npy'; X.KIN=X.ROOT+'/processed_data/kinect_height'; X.OUT=X.ROOT+'/reports/symbolic_v4'
split,data=X.load_all(); NT=X.neural_probs(X.OUT+'/neural_test_oof.csv',data); NV=X.neural_probs(X.OUT+'/neural_val.csv',data)
OUT='/home/claude/v4/fig'; os.makedirs(OUT,exist_ok=True)
C,_=X.candidate_v4(split,data); D1,_=X.candidate_v4(split,data,use_descending=False)
# stacking test preds
r=json.load(open(X.OUT+'/results.json')); W={int(k):np.array(v) for k,v in r['stacking_weights'].items()}
rows={k:[] for k in ('neural','symbolic C','F6 stacking (selected)','F4 gated veto (post-hoc)')}
for f in split['folds']:
    k=f['fold']
    for v in f['test']:
        y=data[v]['y']; pn=NT[v]; s=C[(k,'test',v)]
        rows['neural'].append((v,y,(pn>=.5).astype(int)))
        rows['symbolic C'].append((v,y,s['hard']))
        Xs=np.c_[X.logit(pn),s['p']]; rows['F6 stacking (selected)'].append((v,y,(X.lr_predict(W[k],Xs)>=.5).astype(int)))
        rows['F4 gated veto (post-hoc)'].append((v,y,X.fuse('F4',pn,s)))
# confusion matrices
fig,ax=plt.subplots(1,4,figsize=(16,3.8))
for a,(n,R) in zip(ax,rows.items()):
    m=X.metrics(R); M=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
    a.imshow(M,cmap='Blues'); a.set_title(f"{n}\nMCC {m['MCC']:.3f}",fontsize=10)
    for i in range(2):
        for j in range(2): a.text(j,i,f"{M[i,j]}",ha='center',va='center',color='black' if M[i,j]<M.max()*0.6 else 'white')
    a.set_xticks([0,1],['pred no fall','pred fall']); a.set_yticks([0,1],['no fall','fall'])
plt.tight_layout(); plt.savefig(f'{OUT}/confusion_test.png',dpi=130)
# paired bootstrap over videos for MCC difference vs neural
rng=np.random.default_rng(0); vids=list(range(len(rows['neural'])))
def mcc_of(R,idx):
    y=np.concatenate([R[i][1] for i in idx]); p=np.concatenate([R[i][2] for i in idx]); return S3.mcc(y,p)
diff={}
for n in ('F6 stacking (selected)','F4 gated veto (post-hoc)','symbolic C'):
    d=[]
    for _ in range(2000):
        idx=rng.integers(0,len(vids),len(vids)); d.append(mcc_of(rows[n],idx)-mcc_of(rows['neural'],idx))
    diff[n]=[round(float(np.mean(d)),3),round(float(np.percentile(d,2.5)),3),round(float(np.percentile(d,97.5)),3),round(float(np.mean(np.array(d)>0)),3)]
print('paired bootstrap MCC diff vs neural [mean, lo, hi, P(>0)]',diff)
json.dump(diff,open(f'{OUT}/paired_bootstrap.json','w'))
# error overlap bars (test, neural vs symbolic C)
cp=r['complementarity_test']
fig,a=plt.subplots(figsize=(7,3.2)); labs=['both correct','neural only correct','symbolic only correct','both wrong']
for i,k in enumerate(('A','B','C')):
    vals=[cp[k]['both_correct'],cp[k]['neural_only_correct'],cp[k]['symbolic_only_correct'],cp[k]['both_wrong']]; left=0
    for j,vv in enumerate(vals):
        a.barh(i,vv,left=left,color=['#c9d6e3','#4c78a8','#f58518','#d62728'][j],label=labs[j] if i==0 else None); left+=vv
a.set_yticks([0,1,2],['A original','B v3','C v4']); a.set_xlabel('test frames'); a.legend(fontsize=8,loc='lower right',ncol=2); a.set_title('Neural vs symbolic: frame-level correctness on the test folds',fontsize=10)
plt.tight_layout(); plt.savefig(f'{OUT}/error_overlap.png',dpi=130)
# timelines for 3 test videos
def tl(v,fold,ax):
    h=data[v]['h']; y=data[v]['y']; pn=NT[v]; s=C[(fold,'test',v)]; t=np.arange(len(y))/30
    ax.fill_between(t,0,1.6,where=y==1,color='#fde0dd',label='labelled fall')
    ax.plot(t,h,color='black',lw=1.2,label='neck height (m)'); ax.plot(t,pn*1.6,color='#4c78a8',lw=1,label='neural prob x1.6')
    ax.plot(t,s['hard']*0.08+0.02,color='#f58518',lw=2,label='symbolic fall')
    ax.plot(t,(1-s['upright'])*0.0+X.fuse('F4',pn,s)*0.05+0.12,color='#2ca02c',lw=2,label='F4 output')
    ax.set_title(v+f' (fold {fold})',fontsize=9); ax.set_ylim(0,1.65)
fig,ax=plt.subplots(3,1,figsize=(9,7))
pick=[('fall-03',None),('adl-17',None),('adl-30',None)]
for a,(v,_) in zip(ax,pick):
    fo=[f['fold'] for f in split['folds'] if v in f['test']][0]; tl(v,fo,a)
ax[0].legend(fontsize=7,ncol=5,loc='upper right'); ax[-1].set_xlabel('seconds')
plt.tight_layout(); plt.savefig(f'{OUT}/timelines.png',dpi=120)
# neural error analysis by room (test) and confident mistakes
pf=pd.read_csv(X.OUT+'/per_frame_errors.csv')
g=pf.groupby('room').apply(lambda d: pd.Series(dict(frames=len(d),fall_frames=int(d.label.sum()),neural_FP=int(((d.neural_pred==1)&(d.label==0)).sum()),neural_FN=int(((d.neural_pred==0)&(d.label==1)).sum()))))
print(g)
conf=pf[((pf.neural_prob>=0.9)&(pf.label==0))|((pf.neural_prob<=0.1)&(pf.label==1))]
print('confident neural mistakes',len(conf),'FP>=0.9:',int(((pf.neural_prob>=0.9)&(pf.label==0)).sum()),'FN<=0.1:',int(((pf.neural_prob<=0.1)&(pf.label==1)).sum()))
print('neural FP frames by symbolic state:'); print(pf[(pf.neural_pred==1)&(pf.label==0)].symbolic_state.value_counts())
print('neural FN frames by URFD phase:'); print(pf[(pf.neural_pred==0)&(pf.label==1)].urfd_label.value_counts())
print('neural FN frames by symbolic state:'); print(pf[(pf.neural_pred==0)&(pf.label==1)].symbolic_state.value_counts())
g.to_csv(f'{OUT}/neural_errors_by_room.csv')
