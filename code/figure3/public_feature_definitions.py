"""Numerical image/JSON feature definitions for the OpenStripe extraction entry point."""
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def norm(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def cosine(x):
    x=norm(x);d=np.clip(1-x@x.T,0,2);np.fill_diagonal(d,0);return d
def symmetric(d):
    d=np.maximum((np.asarray(d,dtype=float)+np.asarray(d,dtype=float).T)/2,0);np.fill_diagonal(d,0)
    assert d.shape==(100,100) and np.isfinite(d).all();return d

def foreground_box(a):
    yy,xx=np.where(a<180)
    if len(xx)<32:return (0,0,a.shape[1],a.shape[0])
    return (int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))

def patch_hog(a,x,y,half):
    half=max(12,int(round(half)));pad=np.pad(a,half+2,constant_values=255)
    x=int(round(x))+half+2;y=int(round(y))+half+2
    patch=pad[y-half:y+half+1,x-half:x+half+1]
    patch=cv2.resize(patch,(64,64),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    dx=cv2.Sobel(patch,cv2.CV_32F,1,0);dy=cv2.Sobel(patch,cv2.CV_32F,0,1)
    mag,ang=cv2.cartToPolar(dx,dy);ang=np.mod(ang,np.pi)
    out=[]
    for row in range(4):
        for col in range(4):
            hist=np.histogram(ang[row*16:(row+1)*16,col*16:(col+1)*16],bins=np.linspace(0,np.pi,10),weights=mag[row*16:(row+1)*16,col*16:(col+1)*16])[0]
            out.extend(np.sqrt(hist/max(float(hist.sum()),1e-12)))
    return norm(np.asarray(out))

def spatial_features(points,types,bandwidth):
    grid=np.stack(np.meshgrid(np.linspace(0,1,8),np.linspace(0,1,8)),axis=-1).reshape(-1,2)
    maps=[];counts=[]
    for t in range(4):
        q=points[types==t];counts.append(len(q))
        z=np.exp(-np.sum((grid[:,None,:]-q[None,:,:])**2,axis=-1)/(2*bandwidth**2)).sum(axis=1) if len(q) else np.zeros(64)
        maps.append(norm(z))
    count=norm(np.sqrt(np.log1p(counts)))
    return norm(np.concatenate([np.sqrt(.3)*count,np.sqrt(.7)*norm(np.concatenate(maps))]))

def moments(points,types):
    values=[]
    for t in range(4):
        q=points[types==t]
        values.extend([len(q),int(len(q)>0)])
        values.extend(np.concatenate([q.mean(0),q.std(0),q.min(0),q.max(0)]).tolist() if len(q) else [0.]*8)
    for a in range(4):
        for b in range(a,4):
            qa=points[types==a];qb=points[types==b]
            if len(qa) and len(qb):
                d=cdist(qa,qb)
                if a==b:d=d[np.triu_indices(len(qa),1)]
                else:d=d.ravel()
                values.extend([d.mean(),d.std(),d.min(),d.max()] if d.size else [0.]*4)
            else:values.extend([0.]*4)
    return np.asarray(values)

def shape_context(points,types):
    n=len(points);H=np.zeros((n,4,4,8))
    if n<2:return H.reshape(n,128)
    deltas=points[None,:,:]-points[:,None,:]
    rr=np.linalg.norm(deltas,axis=-1);mean=max(float(rr[~np.eye(n,dtype=bool)].mean()),1e-8)
    theta=np.mod(np.arctan2(deltas[:,:,1],deltas[:,:,0]),2*np.pi)
    rbin=np.clip(np.digitize(rr/mean,[.3,.7,1.3]),0,3)
    abin=np.minimum((theta/(2*np.pi)*8).astype(int),7)
    for i in range(n):
        for j in range(n):
            if i!=j:H[i,types[j],rbin[i,j],abin[i,j]]+=1
    return H.reshape(n,-1)/max(n-1,1)

def pool_typed(desc,offsets,types):
    values=[]
    for i in range(100):
        a,b=offsets[i:i+2];blocks=[]
        for t in range(4):
            v=desc[a:b][types[a:b]==t]
            mean=v.mean(0) if len(v) else np.zeros(desc.shape[1]);sd=v.std(0) if len(v) else np.zeros(desc.shape[1])
            blocks.append(np.concatenate([norm(mean),norm(sd)*.5]))
        values.append(norm(np.concatenate(blocks)))
    return np.stack(values)

def assignment_distance(R,N,mode,penalty,keep_types=None,jitter=None):
    coords=R['canvas'] if mode=='xy_canvas' else R['body']
    if jitter is not None:coords=coords+jitter
    contexts=R['context'];appearance=None
    if mode=='hog_body':appearance=R['hog']
    elif mode=='dino_body':appearance=N['raw']
    elif mode=='dino_white_body':appearance=N['white32']
    distance=np.zeros((100,100))
    for i in range(100):
        ia,ib=R['offsets'][i:i+2]
        for j in range(i+1,100):
            ja,jb=R['offsets'][j:j+2];type_costs=[]
            for t in range(4):
                if keep_types is not None and t not in keep_types:continue
                ai=np.arange(ia,ib)[R['types'][ia:ib]==t];bj=np.arange(ja,jb)[R['types'][ja:jb]==t];n=len(ai);m=len(bj)
                if n+m==0:continue
                if n==0 or m==0:type_costs.append(penalty);continue
                pos=np.minimum(cdist(coords[ai],coords[bj])/.3,1)
                if mode.startswith('xy_'):cost=pos
                else:
                    ca=contexts[ai,None,:];cb=contexts[None,bj,:]
                    ctx=.5*np.sum((ca-cb)**2/np.maximum(ca+cb,1e-12),axis=-1)
                    if appearance is None:cost=.35*pos+.65*ctx
                    else:
                        app=np.clip(1-appearance[ai]@appearance[bj].T,0,2)/2
                        cost=.25*pos+.25*ctx+.5*app
                full=np.full((n+m,n+m),1e6);full[:n,:m]=cost
                full[np.arange(n),m+np.arange(n)]=penalty
                full[n+np.arange(m),np.arange(m)]=penalty;full[n:,m:]=0
                a,b=linear_sum_assignment(full);type_costs.append(float(full[a,b].sum()/max(n,m)))
            distance[i,j]=distance[j,i]=np.mean(type_costs) if type_costs else 0.
    return symmetric(distance)
