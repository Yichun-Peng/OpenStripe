"""Draw Figure 3 panels from numerical inputs using Matplotlib.
The circular tree uses unannotated reconstruction thumbnails."""
from pathlib import Path
import argparse, csv, hashlib, json, os
os.environ.setdefault('MPLCONFIGDIR', str(Path.home()/'.cache/openstripe/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, Wedge
from matplotlib.transforms import Affine2D
from PIL import Image, ImageOps
from scipy.cluster.hierarchy import leaves_list, fcluster
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'data/figure3'
METRICS=['clustering_accuracy','ari','nmi','bcubed_f1']
LABELS=['Accuracy','ARI','NMI','B³ F1']
TYPE_COLORS=dict(zip('BFIC',['#d55e00','#e69f00','#0072b2','#009e73']))
MC=['#0072b2','#d55e00','#009e73','#8b5ca6']
CMAP=LinearSegmentedColormap.from_list('stripeblue',['#f7fbff','#d6eaf4','#95cadd','#39a0c5','#0072b2','#004c7d'])
W=H=180.

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def fmt(v):return f'{0. if abs(v)<.005 else v:.2f}'
def axmm(fig,x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
def txt(fig,x,y,s,**kw):return fig.text(x/W,y/H,s,**kw)
def letter(fig,x,y,s):return txt(fig,x,y,s,fontsize=10,fontweight='bold',va='top',ha='left',bbox=dict(facecolor='white',edgecolor='none',pad=.2),zorder=40)
def framed(ax,grid='y'):
    for sp in ax.spines.values():sp.set_visible(True);sp.set_linewidth(.65);sp.set_color('#24333d')
    ax.tick_params(direction='out',length=2.4,pad=2)
    ax.set_axisbelow(True)
    if grid:ax.grid(axis=grid,color='#e4e9eb',lw=.5)
def load_data(folder):
    d={k:read(folder/n) for k,n in {'matrix':'ablation_17x9_four_metrics.csv','methods':'method_display_order.csv','dose':'category_count_equal_weight_means.csv','gains':'annotation_28_paired_additions.csv','contrib':'annotation_contribution_means.csv','baselines':'baseline_all_50_conditions_recomputed.csv'}.items()}
    d['groups']=[next(r for r in d['matrix'] if int(r['group'])==i) for i in range(1,18)]
    d['best']={r['condition']:r for r in d['matrix'] if r['method']=='Balanced_consensus'}
    d['effects']={r['metric']:r for r in json.loads((folder/'paired_comparison.json').read_text())['metrics']}
    return d

def draw_a(fig,d):
    letter(fig,1.5,179.5,'a')
    headers=['Lei.','kR','KM','FIN','HDB','AHC','D-A','D-kR','Con.']
    methods=[r['method'] for r in d['methods']]
    for k,(metric,name) in enumerate(zip(METRICS,LABELS)):
        x=36+k*36.;a=axmm(fig,x,131.,34.5,40.)
        z=np.array([[float(next(r[metric+'_mean'] for r in d['matrix'] if r['condition']==g['condition'] and r['method']==m)) for m in methods] for g in d['groups']])
        a.imshow(z[:,:8],extent=[-.5,7.5,16.5,-.5],cmap=CMAP,vmin=0,vmax=1,aspect='auto',interpolation='none')
        a.imshow(z[:,8:9],extent=[7.5,9.5,16.5,-.5],cmap=CMAP,vmin=0,vmax=1,aspect='auto',interpolation='none')
        a.set(xlim=(-.5,9.5),ylim=(16.5,-.5),yticks=[])
        a.set_xticks(list(range(8))+[8.5],headers);a.xaxis.tick_top()
        a.tick_params(axis='x',length=0,pad=1,labelsize=5.8,rotation=65)
        a.set_xticks(list(np.arange(-.5,8,1))+[9.5],minor=True);a.set_yticks(np.arange(-.5,17,1),minor=True)
        a.grid(which='minor',color='white',lw=.45);a.tick_params(which='minor',length=0);framed(a,None)
        for i in range(17):a.text(8.5,i,fmt(z[i,8]),fontsize=5.7,ha='center',va='center',color='white' if z[i,8]>=.63 else '#142630')
        a.add_patch(Rectangle((7.5,-.5),2,17,fill=False,edgecolor='#0D4C73',lw=.9))
        a.add_patch(Rectangle((-.5,15.5),10,1,fill=False,edgecolor='#233C4E',lw=.75))
        txt(fig,x+17.25,178.7,name,fontsize=8,ha='center',va='center')
    key=axmm(fig,1.5,131.,32.5,40.);key.set(xlim=(0,32.5),ylim=(16.5,-.5),xticks=[],yticks=[])
    for sp in key.spines.values():sp.set_visible(False)
    for j,t in enumerate('BFIC'):key.text(8.1+j*5.4,-1.25,t,fontsize=7,ha='center',va='center',fontweight='bold')
    for i,g in enumerate(d['groups']):
        cond=g['condition']
        if cond in ['SRC','BW']:
            key.text(16.3,i,'Full-body stripe image' if cond=='SRC' else 'Unannotated stripes',fontsize=6,va='center',ha='center',color='#526573');continue
        for j,t in enumerate('BFIC'):
            enabled=t in cond
            key.scatter(8.1+j*5.4,i,s=10,marker='s',facecolor=TYPE_COLORS[t] if enabled else 'white',edgecolor=TYPE_COLORS[t] if enabled else '#D7DEE2',lw=.5)
    cb=axmm(fig,83,128.7,48,1.)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=CMAP),cax=cb,orientation='horizontal',ticks=[0,.5,1])
    cb.tick_params(labelsize=5.7,length=1,pad=.3)

def draw_b(fig,d):
    a=axmm(fig,8,94,43,31);framed(a);letter(fig,9.1,124.6,'b')
    a.set(xlim=(-.18,4.18),ylim=(0,1.08),xticks=range(5),yticks=[0,.5,1]);a.set_xlabel('Annotation types',labelpad=1.5);a.set_ylabel('Score',labelpad=1.5)
    for k,(metric,name,color) in enumerate(zip(METRICS,LABELS,MC)):
        vals=[float(r[metric+'_mean']) for r in d['dose']]
        a.plot(range(5),vals,color=color,marker=['o','s','^','D'][k],ms=2.8,lw=1,label=name)
        for g in d['groups']:
            if g['condition']!='SRC':a.scatter(int(g['n_types'])+(k-1.5)*.025,float(d['best'][g['condition']][metric+'_mean']),s=6,color=color,alpha=.2,edgecolors='none')
    a.legend(loc='lower right',bbox_to_anchor=(1.005,.01),fontsize=5.8,ncol=2,frameon=False,handlelength=.8,handletextpad=.22,columnspacing=.55,labelspacing=.1,borderaxespad=0)
    return a

def draw_c(fig,d,indices):
    for k in indices:
        metric,name=METRICS[k],LABELS[k];a=axmm(fig,61+k*29.5,94,27.5,31);framed(a,'x')
        a.text(.025,.98,f'c{k+1}',transform=a.transAxes,fontsize=7.4,fontweight='bold',ha='left',va='top',bbox=dict(facecolor='white',edgecolor='none',pad=.2),zorder=40)
        a.text(.98,.97,name,transform=a.transAxes,fontsize=6.9,ha='right',va='top',bbox=dict(facecolor='white',edgecolor='none',pad=.1),zorder=40)
        a.set_ylim(3.55,-1.05);a.set_yticks(range(4),list('BFIC') if k==0 or len(indices)==1 else ['']*4)
        if k==0 or len(indices)==1:a.set_ylabel('Added annotation',labelpad=2)
        a.tick_params(axis='y',length=0,pad=1)
        for i,t in enumerate('BFIC'):
            vals=np.array([float(r[metric+'_gain']) for r in d['gains'] if r['added_type']==t]);assert len(vals)==7
            mean=float(next(r[metric+'_gain_mean'] for r in d['contrib'] if r['added_type']==t))
            a.barh(i,mean,height=.55,color=TYPE_COLORS[t],alpha=.18);a.scatter(vals,i+np.linspace(-.16,.16,7),s=8,color=TYPE_COLORS[t],edgecolors='none');a.plot([mean,mean],[i-.29,i+.29],color='#172D3A',lw=.8)
            a.text(.98,i-.23,f'{mean:.3f}',transform=a.get_yaxis_transform(),fontsize=5.9,ha='right',va='bottom',color=TYPE_COLORS[t],bbox=dict(facecolor='white',edgecolor='none',pad=.05,alpha=.96),zorder=10)
        a.set_xlim(-.09,.66);a.set_xticks([0,.3,.6]);a.tick_params(labelsize=6.5)
        if len(indices)==1:a.set_xlabel('Score increase',labelpad=1.5)
    if len(indices)>1:return txt(fig,119,87,'Score increase',fontsize=7.5,ha='center',va='center')

def draw_d(fig,d,indices):
    others=sorted([r for r in d['baselines'] if r['condition']!='OS_FULL'],key=lambda r:(float(r['ari']),r['condition']))
    osrow=next(r for r in d['baselines'] if r['condition']=='OS_FULL');assert len(others)==49
    xj=.32*((np.arange(49)*.6180339887498949)%1-.5)
    axes=[]
    for k in indices:
        metric,name=METRICS[k],LABELS[k];a=axmm(fig,8+(k%2)*43,49-(k//2)*38,38.5,31);axes.append(a);framed(a)
        a.set(xlim=(-.43,1.4),ylim=(-.045,1.30),xticks=[0,1],xticklabels=['Unannotated\nstripes','OpenStripe'],yticks=[0,.5,1]);a.tick_params(labelsize=6.5)
        if k%2 and len(indices)>1:a.tick_params(axis='y',labelleft=False)
        v=float(osrow[metric]);a.scatter(xj,[float(r[metric]) for r in others],s=8,color='#447B97',alpha=.75,edgecolors='white',lw=.2);a.scatter(1,v,s=24,color='#D55E00',edgecolors='white',lw=.4,zorder=5)
        a.annotate(fmt(v),(1,v),xytext=(0,4),textcoords='offset points',fontsize=7,ha='center',color='#853F11')
        effect=d['effects'][metric];low,high=effect['best_baseline'],effect['OpenStripe']
        a.plot([.35,.44,.44,.35],[low,low,high,high],color='#374B59',lw=.7,zorder=7);a.plot([.18,.35],[low,low],color='#374B59',lw=.5,alpha=.6,zorder=6)
        a.text(.65,(low+high)/2,f"Δ +{effect['delta_best']:.2f}\n{effect['p_display']}",rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=6.2,linespacing=1.35,color='#263B47',zorder=10)
        a.text(.025,.98,f'd{k+1}',transform=a.transAxes,fontsize=8,fontweight='bold',ha='left',va='top',bbox=dict(facecolor='white',edgecolor='none',pad=.2),zorder=40)
        a.text(.63,.045,name,transform=a.transAxes,fontsize=7.4,ha='center',va='bottom',bbox=dict(facecolor='white',edgecolor='none',pad=.1),zorder=40)
    return axes

def tangent(angle):
    deg=(np.degrees(angle)-90+180)%360-180
    if deg>90:deg-=180
    if deg<-90:deg+=180
    return deg

def draw_e(fig,folder,image_dir):
    rows=read(folder/'tree_sample_labels.csv');refs=read(folder/'reference_identity_labels.csv');sids=[r['sample_id'] for r in refs]
    order=np.array(sorted(range(100),key=lambda i:hashlib.sha256(('OpenStripe-blind-order-v1|'+sids[i]).encode()).hexdigest()))
    z=np.load(folder/'automatic_linkage_display_order.npy',allow_pickle=False);auto=np.load(folder/'BFIC_automatic_partition.npz',allow_pickle=False)['auto_K']
    cut=.6110596707818929;partition=np.empty(100,int);partition[order]=fcluster(z,cut,criterion='distance');assert adjusted_rand_score(partition,auto)==1
    seq=[sids[order[i]] for i in leaves_list(z)];assert seq==[r['sample_id'] for r in rows]
    sample={r['sample_id']:r for r in rows};groups=[]
    for s in seq:
        g=sample[s]['automatic_display_cluster']
        if not groups or groups[-1][0]!=g:groups.append((g,[]))
        groups[-1][1].append(s)
    assert len(groups)==29 and sum(r['reference_assignment_differs']=='True' for r in rows)==10
    missing=[s for s in sids if not (image_dir/(s+'.png')).is_file()]
    if missing:raise FileNotFoundError(f'Missing unannotated stripe PNGs in {image_dir}: {missing[:5]}')
    ax=axmm(fig,91.5,2.5,88,83.5);ax.set_aspect('equal');ax.axis('off');ax.set(xlim=(-1.18,1.18),ylim=(-1.18,1.18))
    step=2*np.pi/(100+29*.55);cursor=0;angles={}
    for _,members in groups:
        for s in members:angles[s]=np.pi/2-cursor*step;cursor+=1
        cursor+=.55
    palette=list(plt.get_cmap('tab20').colors)+list(plt.get_cmap('Set2').colors)+list(plt.get_cmap('Set3').colors)
    colors={g:tuple(.84*np.asarray(palette[i])) for i,(g,_) in enumerate(groups)}
    maximum=float(z[:,2].max());radius=lambda h:.95-float(h)/maximum*(.95-.075)
    nodes={i:dict(a=angles[sids[order[i]]],r=.95,members=[sids[order[i]]]) for i in range(100)}
    xy=lambda a,r:(r*np.cos(a),r*np.sin(a))
    for i,row in enumerate(z):
        left,right=nodes[int(row[0])],nodes[int(row[1])];members=left['members']+right['members'];gs={sample[s]['automatic_display_cluster'] for s in members};color=colors[next(iter(gs))] if len(gs)==1 else '#374a56';r=radius(row[2])
        for node in [left,right]:ax.plot(*zip(xy(node['a'],r),xy(node['a'],node['r'])),color=color,lw=.65)
        aa=np.linspace(left['a'],right['a'],50);ax.plot(r*np.cos(aa),r*np.sin(aa),color=color,lw=.65)
        aa=[angles[s] for s in members];nodes[100+i]=dict(a=(min(aa)+max(aa))/2,r=r,members=members)
    for g,members in groups:
        aa=[angles[s] for s in members];lo=min(aa)-step*.38;hi=max(aa)+step*.38;middle=(lo+hi)/2
        ax.add_patch(Wedge((0,0),.983,np.degrees(lo),np.degrees(hi),width=.012,facecolor=colors[g],edgecolor='none'))
        ax.text(*xy(middle,.895),g.replace('C','T',1),fontsize=6.2,fontweight='bold',ha='center',va='center',rotation=tangent(middle),rotation_mode='anchor',color='#21313d',bbox=dict(facecolor='white',edgecolor='none',pad=.1,alpha=.94),zorder=6)
    for s in seq:
        a=angles[s]
        with Image.open(image_dir/(s+'.png')) as im:
            pixels=np.asarray(ImageOps.pad(im.convert('L'),(270,109),color=255).convert('RGBA'))
        width,height=.135,.0545;tr=Affine2D().rotate(np.radians(tangent(a))).translate(*xy(a,1.085))+ax.transData
        art=ax.imshow(pixels,extent=[-width/2,width/2,-height/2,height/2],transform=tr,interpolation='bilinear',zorder=5)
        border=Rectangle((-width/2,-height/2),width,height,transform=tr,facecolor='none',edgecolor='white',lw=.25,zorder=6);ax.add_patch(border);art.set_clip_path(border)
        if sample[s]['reference_assignment_differs']=='True':ax.scatter(*xy(a,.998),s=8,color='#d55e00',edgecolors='white',linewidths=.25,zorder=8)
    ax.add_patch(plt.Circle((0,0),radius(cut),fill=False,color='#bd3846',lw=1,ls=(0,(3,2)),zorder=7))
    return letter(fig,94,81.5,'e')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',default='all',choices=['all','a','b','c','c1','c2','c3','c4','d','d1','d2','d3','d4','e']);p.add_argument('--data-dir',type=Path,default=DEFAULT_DATA);p.add_argument('--image-dir',type=Path,default=ROOT/'data/images_unannotated');p.add_argument('--output-dir',type=Path,default=ROOT/'generated/figure3');p.add_argument('--formats',nargs='+',default=['pdf','png','svg'],choices=['pdf','png','svg']);p.add_argument('--dpi',type=int,default=300);args=p.parse_args()
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    d=load_data(args.data_dir);fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=150);q=args.panel;dose=None;c_label=None;daxes=[];e_label=None
    if q in ['all','a']:draw_a(fig,d)
    if q in ['all','b']:dose=draw_b(fig,d)
    if q in ['all','c'] or q.startswith('c'):c_label=draw_c(fig,d,list(range(4)) if q in ['all','c'] else [int(q[1])-1])
    if q in ['all','d'] or q.startswith('d'):daxes=draw_d(fig,d,list(range(4)) if q in ['all','d'] else [int(q[1])-1])
    if q in ['all','e']:e_label=draw_e(fig,args.data_dir,args.image_dir)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    if dose is not None and c_label is not None:
        bb=dose.xaxis.label.get_window_extent(renderer);cb=c_label.get_window_extent(renderer);c_label.set_y(c_label.get_position()[1]+((bb.y0+bb.y1-cb.y0-cb.y1)/2)/fig.bbox.height)
    if e_label is not None and daxes:
        d1=next(t for t in daxes[0].texts if t.get_text()=='d1');bb=d1.get_window_extent(renderer);eb=e_label.get_window_extent(renderer);e_label.set_y(e_label.get_position()[1]+(bb.y1-eb.y1)/fig.bbox.height)
    args.output_dir.mkdir(parents=True,exist_ok=True);written=[]
    for ext in args.formats:
        target=args.output_dir/f'Figure3_{q}.{ext}';kw={} if q=='all' else {'bbox_inches':'tight','pad_inches':.04}
        fig.savefig(target,dpi=args.dpi,**kw);written.append(str(target.resolve()))
    plt.close(fig);print(json.dumps({'panel':q,'outputs':written,'model_inference':False,'camera_photos_used':False,'annotated_images_used':False,'public_display_difference':'Panel e uses unannotated stripe-reconstruction thumbnails.'},indent=2))
if __name__=='__main__':main()
