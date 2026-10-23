"""Recompute Figure 3 summaries, partition metrics, identity-jackknife tests and consensus from numerical inputs."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import t
from scipy.cluster.hierarchy import linkage,cut_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score,adjusted_rand_score

ROOT=Path(__file__).resolve().parents[2];METRICS=['clustering_accuracy','ari','nmi','bcubed_f1']
def read(path):
    with path.open(encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))
def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def metrics(y,pred,weights=None):
    if weights is None:weights=np.ones(len(y))
    cm=np.bincount(y*(pred.max()+1)+pred,weights=weights,minlength=(y.max()+1)*(pred.max()+1)).reshape(y.max()+1,pred.max()+1)
    cm=cm[cm.sum(1)>0][:,cm.sum(0)>0];n=cm.sum();rows=cm.sum(1);cols=cm.sum(0)
    r,c=linear_sum_assignment(cm,maximize=True);acc=cm[r,c].sum()/n
    pairs=lambda a:(a*(a-1)/2).sum()
    ai,bi,ab,total=pairs(rows),pairs(cols),pairs(cm),n*(n-1)/2;expected=ai*bi/total;den=(ai+bi)/2-expected
    ari=(ab-expected)/den if den else 1.
    probability=cm/n;rr,cc=np.nonzero(cm);mi=np.sum(probability[rr,cc]*np.log(cm[rr,cc]*n/(rows[rr]*cols[cc])))
    entropy=lambda a:-np.sum((a/n)*np.log(a/n));den=(entropy(rows)+entropy(cols))/2;nmi=mi/den if den else 1.
    precision=np.sum(cm*cm/cols[None,:])/n;recall=np.sum(cm*cm/rows[:,None])/n
    return np.array([acc,ari,nmi,2*precision*recall/(precision+recall)])
def internal(distance,labels):
    mask=labels>=0;u,n=np.unique(labels[mask],return_counts=True)
    if len(u)<2 or len(u)>=mask.sum():return -1e9
    return float(silhouette_score(distance[np.ix_(mask,mask)],labels[mask],metric='precomputed'))*(.5+.5*mask.mean())-.15*np.mean(n==1)-.1*max(0.,n.max()/mask.sum()-.65)
def consensus(data,out):
    config=json.loads((data/'consensus_sources.json').read_text());bank=np.load(data/'BFIC_consensus_source_partitions.npz',allow_pickle=False);group_means=[]
    for group in config['sources']:
        mats=[]
        for code in group['conditions']:
            labels=bank[code+'__'+group['method']].copy();start=int(labels[labels>=0].max()+1) if (labels>=0).any() else 0
            for i in np.flatnonzero(labels<0):labels[i]=start;start+=1
            mats.append((labels[:,None]==labels[None,:]).astype(float))
        group_means.append(np.mean(mats,axis=0))
    distance=1-np.mean(group_means,axis=0);np.fill_diagonal(distance,0)
    expected=np.load(data/'BFIC_consensus_distance.npy',allow_pickle=False);np.testing.assert_allclose(distance,expected,rtol=0,atol=1e-12)
    refs=read(data/'reference_identity_labels.csv');sid=[r['sample_id'] for r in refs];y=np.array([int(r['reference_identity'][1:])-1 for r in refs])
    order=np.array(sorted(range(100),key=lambda i:hashlib.sha256(('OpenStripe-blind-order-v1|'+sid[i]).encode()).hexdigest()))
    ordered=distance[np.ix_(order,order)];z=linkage(squareform(ordered,checks=False),method='average');ks=np.arange(2,61);labs=cut_tree(z,n_clusters=ks);scores=np.array([internal(ordered,labs[:,i]) for i in range(len(ks))]);best=int(np.argmax(scores));prediction=np.empty(100,int);prediction[order]=labs[:,best]
    frozen=np.load(data/'BFIC_automatic_partition.npz',allow_pickle=False)['auto_K'];assert adjusted_rand_score(prediction,frozen)==1
    np.save(out/'BFIC_consensus_distance.npy',distance);np.save(out/'consensus_linkage_blind_order.npy',z);np.savez_compressed(out/'BFIC_refitted_partition.npz',auto_K=prediction)
    return {'source_groups':len(group_means),'source_partitions_available':len(bank.files),'selected_K':int(ks[best]),'internal_score':float(scores[best]),'agreement_ARI_to_archived_partition':float(adjusted_rand_score(prediction,frozen)),'metrics':dict(zip(METRICS,map(float,metrics(y,prediction))))}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=ROOT/'data/figure3');p.add_argument('--output-dir',type=Path,default=ROOT/'generated/figure3_reanalysis');p.add_argument('--refit-consensus',action='store_true');p.add_argument('--bootstrap',type=int,default=0,help='Optional fixed-partition identity bootstrap, e.g. 1999; not used for displayed p values.');args=p.parse_args();d=args.data_dir;out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    matrix=read(d/'ablation_17x9_four_metrics.csv');runs=read(d/'ablation_all_runs_recomputed.csv');rebuilt=[];max_abs=0
    for row in matrix:
        rr=[r for r in runs if r['condition']==row['condition'] and r['method']==row['method']];assert len(rr)==int(row['n_runs']);new=dict(row)
        for m in METRICS:
            vals=np.array([float(r[m]) for r in rr]);mean=float(vals.mean());max_abs=max(max_abs,abs(mean-float(row[m+'_mean'])));np.testing.assert_allclose(mean,float(row[m+'_mean']),atol=1e-12)
            new[m+'_mean']=mean;new[m+'_sd']=float(vals.std(ddof=1)) if len(vals)>1 else ''
        rebuilt.append(new)
    write(out/'ablation_17x9_four_metrics.csv',rebuilt)
    best={r['condition']:r for r in rebuilt if r['method']=='Balanced_consensus'};dose=[]
    for row in read(d/'category_count_equal_weight_means.csv'):
        new=dict(row);n=int(row['n_types']);subset=[r for r in best.values() if r['condition']!='SRC' and int(r['n_types'])==n]
        for m in METRICS:
            v=float(np.mean([r[m+'_mean'] for r in subset]));np.testing.assert_allclose(v,float(row[m+'_mean']),atol=1e-12);new[m+'_mean']=v
        dose.append(new)
    write(out/'category_count_equal_weight_means.csv',dose)
    additions=[]
    for row in read(d/'annotation_28_paired_additions.csv'):
        new=dict(row)
        for m in METRICS:
            v=float(best[row['larger_subset']][m+'_mean']-best[row['background']][m+'_mean']);np.testing.assert_allclose(v,float(row[m+'_gain']),atol=1e-12);new[m+'_gain']=v
        additions.append(new)
    write(out/'annotation_28_paired_additions.csv',additions);contrib=[]
    for row in read(d/'annotation_contribution_means.csv'):
        new=dict(row)
        for m in METRICS:
            v=float(np.mean([r[m+'_gain'] for r in additions if r['added_type']==row['added_type']]));np.testing.assert_allclose(v,float(row[m+'_gain_mean']),atol=1e-12);new[m+'_gain_mean']=v
        contrib.append(new)
    write(out/'annotation_contribution_means.csv',contrib)
    saved=np.load(d/'paired_comparison_inputs.npz',allow_pickle=False);y,pred,conditions=saved['reference'],saved['predictions'],saved['conditions'];assert pred.shape==(50,100) and len(np.unique(y))==28 and conditions[0]=='OS_FULL'
    observed=np.array([metrics(y,x) for x in pred]);rows=read(d/'baseline_all_50_conditions_recomputed.csv');baseline=[]
    for i,c in enumerate(conditions):
        row=next(r for r in rows if r['condition']==str(c));new=dict(row)
        for k,m in enumerate(METRICS):np.testing.assert_allclose(observed[i,k],float(row[m]),atol=1e-12);new[m]=float(observed[i,k])
        baseline.append(new)
    write(out/'baseline_all_50_conditions_recomputed.csv',baseline)
    delta=observed[0]-observed[1:].max(0);deleted=[]
    for identity in np.unique(y):
        scores=np.array([metrics(y,x,(y!=identity).astype(int)) for x in pred]);deleted.append(scores[0]-scores[1:].max(0))
    deleted=np.array(deleted);np.testing.assert_allclose(deleted,np.load(d/'paired_identity_jackknife.npy'),atol=1e-12);n=len(deleted);se=np.sqrt((n-1)/n*((deleted-deleted.mean(0))**2).sum(0));stat=delta/se;pval=2*t.sf(abs(stat),n-1)
    result=json.loads((d/'paired_comparison.json').read_text());result.pop('bootstrap_role',None)
    for k,row in enumerate(result['metrics']):
        np.testing.assert_allclose([row['p_raw'],row['delta_best'],row['jackknife_se']],[pval[k],delta[k],se[k]],atol=1e-12)
        lo,hi=delta[k]+np.array([-1,1])*t.ppf(.975,n-1)*se[k]
        row.update(OpenStripe=float(observed[0,k]),best_baseline=float(observed[1:,k].max()),delta_best=float(delta[k]),jackknife_se=float(se[k]),t_statistic=float(stat[k]),degrees_of_freedom=n-1,p_raw=float(pval[k]),ci_low=float(lo),ci_high=float(hi));row.pop('p_holm4_audit',None);row.pop('bootstrap_diagnostic_ci',None)
    if args.bootstrap:
        rng=np.random.default_rng(20261015);boot=np.empty((args.bootstrap,4));ids=np.unique(y)
        for b in range(args.bootstrap):
            count=np.bincount(rng.choice(ids,len(ids),replace=True),minlength=int(y.max())+1);scores=np.array([metrics(y,x,count[y]) for x in pred]);boot[b]=scores[0]-scores[1:].max(0)
        np.save(out/'paired_bootstrap_differences.npy',boot);ci=np.quantile(boot,[.025,.975],axis=0)
        result['bootstrap_diagnostic']={'replicates':args.bootstrap,'seed':20261015,'intervals':{m:list(map(float,ci[:,k])) for k,m in enumerate(METRICS)}}
    dump(out/'paired_comparison.json',result);np.save(out/'paired_identity_jackknife.npy',deleted)
    report={'status':'PASS','heatmap_values_verified':612,'paired_addition_points_verified':112,'baseline_metric_values_verified':200,'maximum_heatmap_difference':max_abs,'jackknife_identity_units':n,'displayed_p':'raw two-sided approximate t using paired delete-one-identity jackknife SE; df=27','model_inference_performed':False,'photographs_used':False,'annotated_images_used':False,'scope':'Partition-based summaries, metrics, jackknife tests and consensus refitting.'}
    if args.refit_consensus:report['consensus_backend_refit']=consensus(d,out)
    for filename in ['method_display_order.csv','reference_identity_labels.csv','tree_sample_labels.csv','automatic_linkage_display_order.npy','BFIC_automatic_partition.npz']:
        shutil.copy2(d/filename,out/filename)
    dump(out/'verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
