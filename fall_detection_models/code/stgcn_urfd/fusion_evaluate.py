import json,pickle,sys,numpy as np,pandas as pd
sys.path.insert(0,'.'); import symbolic_v2 as V
B='/home/claude/audit/bundle'; sp=json.load(open(B+'/splits/split_scene_v2.json'))
F=pickle.load(open('/home/claude/sym/feats.pkl','rb'))
cols=['v','f','lab','hw','mm','occ','std','hh','H','Dmm','p40']
K=pd.read_csv(B+'/URFD/features/urfall-cam0-falls.csv',header=None,names=cols)
def lying_h(vids):
    L=[]
    for v in vids:
        if v[0]!='f': continue
        h=np.nan_to_num(F[v][0],nan=1.0); i=K[(K.v==v)&(K.lab==1)].f.values-1; L.append(h[i[i<len(h)]])
    return np.concatenate(L)
def events(rows):
    det=falls=clean=adl=eps=fr=0
    for v,y,p in rows:
        if v[0]=='f': falls+=1; det+=int(p[int(np.argmax(y)):].any())
        else: adl+=1; clean+=int(not p.any()); fr+=len(p); eps+=int(p[0])+int((np.diff(p)==1).sum())
    return dict(falls=f'{det}/{falls}',adl_clean=f'{clean}/{adl}',fa_min=round(eps/(fr/1800),2),_d=det,_c=clean)
def metrics(rows):
    y=np.concatenate([r[1] for r in rows]); p=np.concatenate([r[2] for r in rows])
    tp=((y==1)&(p==1)).sum(); fp=((y==0)&(p==1)).sum(); fn=((y==1)&(p==0)).sum(); tn=((y==0)&(p==0)).sum()
    pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
    return dict(MCC=round(V.mcc(y,p),3),F1=round(2*pr*rc/max(pr+rc,1e-9),3),Prec=round(pr,3),Rec=round(rc,3),
                Spec=round(tn/max(tn+fp,1),3),Acc=round((tp+tn)/len(y),3),**events(rows))
def probs(df,v,T):
    d=df[df.sequence==v].sort_values('frame'); p=np.zeros(T); p[d.frame.values-1]=d.prob.values; return p
part=sys.argv[1]   # val | test
ens=pd.read_csv('/home/claude/sym/urfd_val_ens.csv' if part=='val' else '/tmp/urfd_oof.csv')
N,Fu,S,Vt=[],[],[],[]
for f in sp['folds']:
    m,tm=V.fit([F[v] for v in f['train']], lying_h(f['train']))
    print(f"fold {f['fold']}: tau_h={m.tau_h:.3f} tau_v={m.tau_v:.3f} tau_s={m.tau_s:.3f} train trigger MCC={tm:.3f}")
    for v in f[part]:
        h,vz,y=F[v]; s=m.predict(h,vz); p=probs(ens,v,len(y))
        N.append((v,y,(p>=0.5).astype(int))); Fu.append((v,y,V.fuse(p,s))); S.append((v,y,s['trigger'])); Vt.append((v,y,((p>=0.5)&(V.support_window(s['state'])==1)).astype(int)))
for name,rows in (('N  neural ensemble',N),('S  symbolic trigger',S),('F  fused Eq.11-13',Fu),('V  veto only',Vt)):
    print(part,name,{k:v for k,v in metrics(rows).items() if k[0]!='_'})
if part=='val':
    a,b=metrics(N),metrics(Fu)
    ok=b['MCC']>=a['MCC']+0.02 and b['_d']>=a['_d'] and b['_c']>=a['_c']
    print('DECISION F:', 'ADOPT' if ok else 'REJECT')
    c=metrics(Vt); ok=c['MCC']>=a['MCC']+0.02 and c['_d']>=a['_d'] and c['_c']>=a['_c']
    print('DECISION V:', 'ADOPT' if ok else 'REJECT')
