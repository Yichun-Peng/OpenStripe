"""Compute geometry, typed shape contexts, HOG, occupancy, moments and distances from BW PNGs and JSONs.
Optional DINOv2 inference accepts a hash-checked checkpoint through --dino-weights."""
from pathlib import Path
import argparse,csv,hashlib,json,time,os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'1')
os.environ.setdefault('TIMM_FUSED_ATTN','0')
import numpy as np
from PIL import Image
from public_feature_definitions import norm,cosine,foreground_box,patch_hog,shape_context,spatial_features,moments,pool_typed,assignment_distance
ROOT=Path(__file__).resolve().parents[2]
TYPES=['breakpoint','bifurcation','irregular','round']
DINO_SHA='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def load_inputs(root):
    with (root/'data/samples.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==100 and len({r['sample_id'] for r in rows})==100
    # Use only sample correspondence and relative file names; identity is not an input to features.
    return [{'sample_id':r['sample_id'],'image':root/r['common_bw_image'],'annotation':root/r['annotation']} for r in rows]
def build_points(rows):
    body=[];canvas=[];types=[];hog=[];contexts=[];offsets=[0];boxes=[];counts=dict.fromkeys(TYPES,0)
    for row in rows:
        metadata=json.loads(row['annotation'].read_text());assert metadata['sample_id']==row['sample_id'];assert sha(row['image'])==metadata['image']['sha256']
        a=np.asarray(Image.open(row['image']).convert('L'));height,width=a.shape
        assert [width,height]==[metadata['image']['width'],metadata['image']['height']]
        box=foreground_box(a);assert list(box)==metadata['coordinate_system']['body_bbox_xyxy']
        x0,y0,x1,y1=box;bw,bh=x1-x0,y1-y0
        markers=metadata['annotations'];p=np.asarray([[q['x_pixel'],q['y_pixel']] for q in markers],dtype=float).reshape(-1,2);t=np.array([TYPES.index(q['marker_type']) for q in markers],dtype=int)
        pc=p/np.array([width,height]);pb=(p-np.array([x0,y0]))/np.array([bw,bh]);body.append(pb);canvas.append(pc);types.append(t)
        for q in markers:counts[q['marker_type']]+=1
        diag=np.hypot(bw,bh);hog.extend([norm(np.concatenate([patch_hog(a,x,y,diag*f/2) for f in [.08,.16,.24]])) for x,y in p])
        contexts.append(shape_context(pb,t));offsets.append(offsets[-1]+len(p));boxes.append(box)
    result={'body':np.vstack(body),'canvas':np.vstack(canvas),'types':np.concatenate(types),'hog':np.vstack(hog),'context':np.vstack(contexts),'offsets':np.array(offsets),'boxes':np.array(boxes)}
    assert len(result['body'])==316 and list(counts.values())==[46,140,37,93]
    return result,counts

def neural_points(rows,R,checkpoint,device='cpu'):
    if sha(checkpoint)!=DINO_SHA:raise ValueError('DINOv2 checkpoint SHA256 differs from the archived public ViT-S/14 checkpoint.')
    import torch,timm
    from sklearn.decomposition import PCA
    torch.set_num_threads(4)
    model=timm.create_model('vit_small_patch14_dinov2',pretrained=False,num_classes=0,dynamic_img_size=True)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);state.pop('mask_token',None);model.load_state_dict(state,strict=True);model.eval().to(device)
    for parameter in model.parameters():parameter.requires_grad=False
    values=[];glob=[];size=518;xx,yy=np.meshgrid(np.arange(37)+.5,np.arange(37)+.5);grid=np.stack([xx,yy],axis=-1).reshape(-1,2)
    mean=torch.tensor([.485,.456,.406])[:,None,None];sd=torch.tensor([.229,.224,.225])[:,None,None]
    for i,row in enumerate(rows):
        x0,y0,x1,y1=R['boxes'][i];im=Image.open(row['image']).convert('RGB').crop((x0,y0,x1,y1));im.thumbnail((size,size),Image.Resampling.BICUBIC);rw,rh=im.size;ox=(size-rw)//2;oy=(size-rh)//2
        padded=Image.new('RGB',(size,size),'white');padded.paste(im,(ox,oy));arr=np.asarray(padded).copy();tensor=torch.from_numpy(arr).permute(2,0,1).float()/255;tensor=((tensor-mean)/sd).to(device)
        with torch.inference_mode():z=model.forward_features(tensor[None])[0,model.num_prefix_tokens:].cpu().numpy()
        assert z.shape==(1369,384);glob.append(norm(z.mean(0)));a,b=R['offsets'][i:i+2]
        for point in R['body'][a:b]:
            center=(point*np.array([rw,rh])+np.array([ox,oy]))/14;d2=((grid-center)**2).sum(1);blocks=[]
            for sigma in [1.,3.]:
                weights=np.exp(-d2/(2*sigma*sigma));blocks.append(norm(weights@z/max(weights.sum(),1e-12)))
            values.append(norm(np.concatenate(blocks)))
        if (i+1)%10==0:print(f'DINOv2 public BW inference: {i+1}/100',flush=True)
    raw=np.vstack(values);pca=PCA(n_components=32,whiten=True,svd_solver='full');white=norm(pca.fit_transform(raw))
    return {'raw':raw,'white32':white,'global':np.vstack(glob)},{'components':pca.components_,'mean':pca.mean_,'explained_variance':pca.explained_variance_,'singular_values':pca.singular_values_}

def compare_npz(actual,path):
    expected=np.load(path,allow_pickle=False);checks=[]
    for key,value in actual.items():
        other=expected[key];assert value.shape==other.shape,(key,value.shape,other.shape)
        maximum=float(np.max(abs(value-other))) if value.size else 0.;ok=bool(np.allclose(value,other,rtol=1e-10,atol=1e-10))
        checks.append({'array':key,'shape':list(value.shape),'maximum_abs_difference':maximum,'matches':ok})
    return checks

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release-root',type=Path,default=ROOT);p.add_argument('--output-dir',type=Path,default=ROOT/'generated/figure3_features');p.add_argument('--reference-dir',type=Path,default=ROOT/'data/figure3/reference_features');p.add_argument('--skip-reference-comparison',action='store_true');p.add_argument('--assignment-distances',action='store_true',help='Also recompute non-neural typed Hungarian assignment distances.');p.add_argument('--dino-weights',type=Path,help='Explicit official dinov2_vits14_pretrain.pth; full inference on all 100 public BW images.');p.add_argument('--device',default='cpu',choices=['cpu','cuda','mps']);args=p.parse_args();start=time.time();out=args.output_dir;out.mkdir(parents=True,exist_ok=True);rows=load_inputs(args.release_root);R,counts=build_points(rows);np.savez_compressed(out/'point_geometry_hog.npz',**R)
    G={}
    for name,coord,bw in [('soft_canvas_008','canvas',.08),('soft_body_004','body',.04),('soft_body_008','body',.08),('soft_body_012','body',.12)]:G[name]=np.vstack([spatial_features(R[coord][a:b],R['types'][a:b],bw) for a,b in zip(R['offsets'][:-1],R['offsets'][1:])])
    m=np.vstack([moments(R['body'][a:b],R['types'][a:b]) for a,b in zip(R['offsets'][:-1],R['offsets'][1:])]);scale=np.sqrt(np.mean(m*m,axis=0));scale[scale<1e-8]=1;G['moments_body']=norm(m/scale);G['local_hog']=pool_typed(R['hog'],R['offsets'],R['types']);np.save(out/'moments_rms_scale.npy',scale)
    N={};neural_report=None
    if args.dino_weights:
        N,pca=neural_points(rows,R,args.dino_weights,args.device);np.savez_compressed(out/'local_dino.npz',**N);np.savez_compressed(out/'local_patch_pca32.npz',**pca);G['local_dino_raw']=pool_typed(N['raw'],R['offsets'],R['types']);G['local_dino_white32']=pool_typed(N['white32'],R['offsets'],R['types']);neural_report={'model':'DINOv2 ViT-S/14','checkpoint_sha256':DINO_SHA,'inference_performed':True,'images':100,'device':args.device}
    np.savez_compressed(out/'marker_blocks.npz',**G);distances={}
    for name,feature in G.items():
        distances['G_'+name]=cosine(feature);np.save(out/('G_'+name+'.npy'),distances['G_'+name])
    if args.assignment_distances:
        for mode in ['xy_canvas','xy_body','shape_body','hog_body']+(['dino_body','dino_white_body'] if N else []):
            for penalty in [.25,.5]:
                name=f'M_{mode}_{int(penalty*100)}';distances[name]=assignment_distance(R,N,mode,penalty);np.save(out/(name+'.npy'),distances[name])
    report={'status':'PASS','input_mode':'Public unannotated black-and-white PNG plus per-image annotation JSON; no marked PNG parser.','images':100,'points':316,'counts':counts,'point_arrays':{k:list(v.shape) for k,v in R.items()},'computed_blocks':{k:list(v.shape) for k,v in G.items()},'computed_distances':list(distances),'identity_labels_used_for_features':False,'raw_camera_photos_read':False,'annotated_images_read':False,'neural_inference':neural_report or {'inference_performed':False,'reason':'Optional DINOv2 checkpoint not supplied.'},'full_49_baseline_encoders_rerun':False,'elapsed_seconds':round(time.time()-start,2)}
    if not args.skip_reference_comparison:
        if not (args.reference_dir/'point_geometry_hog.npz').exists():raise FileNotFoundError('Reference arrays absent; supply --reference-dir or --skip-reference-comparison.')
        report['point_reference_comparison']=compare_npz(R,args.reference_dir/'point_geometry_hog.npz');checks=list(report['point_reference_comparison']);report['distance_reference_comparison']=[]
        for name,mat in distances.items():
            target=args.reference_dir/(name+'.npy')
            if target.exists():
                other=np.load(target,allow_pickle=False);error=float(np.max(abs(mat-other)));ok=bool(np.allclose(mat,other,rtol=1e-10,atol=1e-10));report['distance_reference_comparison'].append({'condition':name,'maximum_abs_difference':error,'matches':ok});checks.append({'matches':ok})
        if N and (args.reference_dir/'local_dino.npz').exists():
            report['neural_reference_comparison']=compare_npz(N,args.reference_dir/'local_dino.npz');checks+=report['neural_reference_comparison']
        report['status']='PASS' if all(r['matches'] for r in checks) else 'DIFFERENCE'
    dump(out/'extraction_verification.json',report);print(json.dumps(report,indent=2))
    if report['status']!='PASS':raise SystemExit('Computed features differ from archived references; differences are reported, not silently substituted.')
if __name__=='__main__':main()
