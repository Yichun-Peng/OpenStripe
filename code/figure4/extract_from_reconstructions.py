"""Extract 18 traits from native SRC images using torso ROIs, Otsu thresholding and 15-pixel speckle removal.
Export measurements, core-eight Ledoit-Wolf distances and Pixel IoU distances."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from sklearn.covariance import LedoitWolf

from features import FEATURE_NAMES,extract_features
from preprocess import preprocess_image,canonical_mask

ROOT=Path(__file__).resolve().parents[2]
CORE8=['fractal_dimension','rel_spacing_median','dv_density_gradient',
       'melanistic_ratio','dominant_orientation_sin2','width_iqr_ratio',
       'bifurcation_rate','isolated_spot_fraction']


def pixel_iou(masks,image_ids,ids):
    # Float32 matches the original zero-shift all-cross-image comparison.
    vectors=np.asarray([m.ravel() for m in masks],dtype=np.float32)
    intersection=vectors@vectors.T;area=vectors.sum(1)
    difference=1-intersection/(area[:,None]+area[None,:]-intersection)
    indices=[np.flatnonzero(np.asarray(image_ids)==i) for i in ids]
    result=np.zeros((len(ids),len(ids)))
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            result[i,j]=result[j,i]=float(difference[np.ix_(indices[i],indices[j])].mean())
    return result


def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    rois=pd.read_csv(args.rois,dtype={'sample_id':str,'individual_id':str})
    rows=[];masks=[];image_ids=[];qc_rows=[]
    for row in rois.itertuples():
        path=args.images/(row.sample_id+'.png')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=row.source_reconstruction_sha256:
            raise ValueError(f'{row.sample_id}: source-image SHA256 differs from recorded morphology input')
        roi=tuple(int(getattr(row,k)) for k in ['source_roi_x0','source_roi_y0','source_roi_x1','source_roi_y1'])
        mask,qc=preprocess_image(path,min_speckle_area=15,flip_right=False,flank='unknown',roi=roi)
        values,diag=extract_features(mask)
        if not diag['valid']:raise ValueError(f'{row.sample_id}: '+str(diag['invalid_reason']))
        rows.append(dict(individual_id=row.individual_id,sample_id=row.sample_id,**values))
        qc_rows.append(dict(sample_id=row.sample_id,individual_id=row.individual_id,
            threshold=qc['threshold'],removed_pixels=qc['removed_pixels'],
            torso_length_px=qc['torso_length_px'],torso_height_px=qc['torso_height_px']))
        masks.append(canonical_mask(mask,(256,512)));image_ids.append(row.individual_id)
        print(row.sample_id,flush=True)
    per_image=pd.DataFrame(rows)
    identity=per_image.groupby('individual_id',sort=True)[list(FEATURE_NAMES)].mean()
    # Original preprocessing: finite-column median imputation, population SD.
    X=identity.copy()
    for col in X:
        a=X[col].to_numpy(copy=True);a[~np.isfinite(a)]=np.median(a[np.isfinite(a)])
        X[col]=a
    Z=(X-X.mean())/X.std(ddof=0)
    fit=LedoitWolf().fit(Z[CORE8].to_numpy())
    D=squareform(pdist(Z[CORE8],metric='mahalanobis',VI=fit.precision_))
    iou=pixel_iou(masks,image_ids,identity.index.tolist())
    per_image.to_csv(out/'per_image_traits18.csv',index=False)
    identity.to_csv(out/'identity_traits18.csv',index_label='individual_id')
    Z.to_csv(out/'standardized_traits18.csv',index_label='individual_id')
    pd.DataFrame(qc_rows).to_csv(out/'preprocessing.csv',index=False)
    np.savez_compressed(out/'extracted_matrices.npz',identity_ids=identity.index.to_numpy(dtype=str),
        morph_mahalanobis=D,pixel_iou=iou)
    report={'status':'PASS','image_records':len(rows),'identities':len(identity),
        'features':list(FEATURE_NAMES),'core8':CORE8,'LedoitWolf_shrinkage':float(fit.shrinkage_),
        'uses_annotation_overlays':False,'image_normalization':'full-resolution source ROI then tight stripe-support bbox',
        'canonical_pixel_grid':[256,512]}
    if args.verify:
        ref=pd.read_csv(args.reference/'per_image_traits18.csv',float_precision='round_trip')
        merged=per_image.merge(ref,on=['individual_id','sample_id'],suffixes=('_new','_ref'),validate='one_to_one')
        errors={f:float(np.nanmax(abs(merged[f+'_new']-merged[f+'_ref']))) for f in FEATURE_NAMES}
        report['maximum_absolute_error']=errors
        report['matches_frozen_measurements']=all(e<=1e-8 for e in errors.values())
        if not report['matches_frozen_measurements']:report['status']='NUMERICAL_REVIEW'
    (out/'extraction_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,default=ROOT/'data/images_unannotated_src')
    p.add_argument('--rois',type=Path,default=ROOT/'data/figure4/morphology_rois.csv')
    p.add_argument('--out',type=Path,default=ROOT/'build/figure4/extracted')
    p.add_argument('--verify',action='store_true')
    p.add_argument('--reference',type=Path,default=ROOT/'data/figure4')
    run(p.parse_args())
