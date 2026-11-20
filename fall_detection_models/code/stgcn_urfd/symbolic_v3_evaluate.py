import json,pickle,sys,numpy as np,pandas as pd
sys.path.insert(0,'.'); import symbolic_v2 as V
exec(open('fusion_evaluate.py').read().split("part=sys.argv[1]")[0])   # helpers + v2 feats F
KH='../../../processed_data/kinect_height'
F3={}
for v,(h2,_,y) in F.items():
    h=np.load(f'{KH}/{v}.npz')['height_m'].astype(float); h=V._ffill(h)
    F3[v]=(h,V.descent_speed(h),y)
def lying_h3(vids):
    L=[]
    for v in vids:
        if v[0]!='f': continue
        h=np.nan_to_num(F3[v][0],nan=2.0); i=K[(K.v==v)&(K.lab==1)].f.values-1; L.append(h[i[i<len(h)]])
    return np.concatenate(L)
HG=np.linspace(0.10,0.80,15); VG=np.linspace(0.25,2.5,15)
res={}
for name,FF,ly,grid in (('v2',F,lying_h,(np.linspace(0.10,0.80,15),np.linspace(0.25,2.0,15))),('v3',F3,lying_h3,(HG,VG))):
    rows={('val','trigger'):[],('val','state'):[],('test','trigger'):[],('test','state'):[]}
    for f in sp['folds']:
        if name=='v3':
            # metric heights: NaN means unknown -> treat as high (2 m) inside predict via nan_to_num(1.0)? use 2.0
            pass
        m,tm=V.fit([FF[v] for v in f['train']], ly(f['train']), *grid)
        print(name,f"fold {f['fold']}: tau_h={m.tau_h:.3f} tau_v={m.tau_v:.3f} tau_s={m.tau_s:.3f} train MCC={tm:.3f}")
        for part in ('val','test'):
            for v in f[part]:
                h,vz,y=FF[v]; s=m.predict(h,vz)
                rows[(part,'trigger')].append((v,y,s['trigger'])); rows[(part,'state')].append((v,y,s['state']))
    res[name]=rows
    for part in ('val',):
        for k in ('trigger','state'):
            print(name,part,k,{a:b for a,b in metrics(rows[(part,k)]).items() if a[0]!='_'})
a=metrics(res['v2'][('val','trigger')]); b=metrics(res['v3'][('val','trigger')])
ok=b['MCC']>a['MCC'] and b['_d']>=a['_d']; print('DECISION:', 'v3 replaces v2' if ok else 'keep v2')
best='v3' if ok else 'v2'
for k in ('trigger','state'):
    print('TEST',best,k,{x:y for x,y in metrics(res[best][('test',k)]).items() if x[0]!='_'})
pickle.dump(res,open('res_v3.pkl','wb'))
