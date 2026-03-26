import os, re, io, json, zipfile, warnings
from typing import Dict, List, Optional
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
VAR_EPS=1e-12; MICROARRAY_RANGE_MAX=20.0

# ---------- IO ----------
def _as_bio(x):
    if isinstance(x, io.BytesIO): x.seek(0); return x
    if isinstance(x, bytes): return io.BytesIO(x)
    return None

def _ext_hint(x,h):
    if h: return os.path.splitext(str(h))[1].lower()
    if isinstance(x,(str,os.PathLike)): return os.path.splitext(str(x))[1].lower()
    return ""

def read_table(x,hint=None,assume_first_gene=True,group_name=None)->pd.DataFrame:
    is_path=isinstance(x,(str,os.PathLike)); ext=_ext_hint(x,hint)
    def r_xl(y): return pd.read_excel(y if is_path else _as_bio(y), engine="openpyxl")
    def r_csv(y,sep=None):
        if sep is None: return pd.read_csv(y if is_path else _as_bio(y), sep=sep, engine="python")
        return pd.read_csv(y if is_path else _as_bio(y), sep=sep)
    if ext in (".xlsx",".xls"): df=r_xl(x)
    elif ext in (".tsv",".txt"): df=r_csv(x,"\t")
    elif ext==".csv": df=r_csv(x,None)
    else:
        try: df=r_xl(x)
        except: df=r_csv(x,None)
    df=df.dropna(how="all").dropna(axis=1,how="all")
    lower=[str(c).strip().lower() for c in df.columns]; gene_col=None
    for k in ["biomarkers","biomarker","marker","gene","feature","id","name"]:
        if k in lower: gene_col=df.columns[lower.index(k)]; break
    if gene_col is None:
        if not assume_first_gene: raise ValueError("Gene column not found.")
        gene_col=df.columns[0]
    df=df.rename(columns={gene_col:"Biomarker"}).set_index("Biomarker")
    for c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(axis=1,how="all")
    if group_name: df.columns=[f"{group_name}__{c}" for c in df.columns]
    df.index=(df.index.astype(str).str.strip().str.upper().str.replace(r"\.\d+$","",regex=True))
    return df.groupby(level=0).median(numeric_only=True)

def read_meta(x,hint=None)->pd.DataFrame:
    is_path=isinstance(x,(str,os.PathLike)); ext=_ext_hint(x,hint)
    if ext in (".xlsx",".xls"): return pd.read_excel(x if is_path else _as_bio(x), engine="openpyxl")
    try: return pd.read_csv(x if is_path else _as_bio(x), sep=None, engine="python")
    except:
        for s in [",","\t",";","|"]:
            try: return pd.read_csv(x if is_path else _as_bio(x), sep=s)
            except: pass
    return pd.read_excel(x if is_path else _as_bio(x), engine="openpyxl")

# ---------- helpers ----------
def normalize_group_value(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return None
    return str(x).strip()

def build_meta_from_cols(cols:List[str],use_prefix=True)->pd.DataFrame:
    if use_prefix and any("__" in c for c in cols):
        return pd.DataFrame({"sample":cols,"group":[c.split("__",1)[0] if "__" in c else "Unknown" for c in cols],
                             "bare_id":[c.split("__",1)[-1] for c in cols]}).set_index("sample")
    return pd.DataFrame({"sample":cols,"group":"ALL","bare_id":cols}).set_index("sample")

def _guess_batch_col(df:pd.DataFrame)->Optional[str]:
    for k in ["batch","Batch","run","Run","lane","Lane","plate","Plate","flowcell","Flowcell","center","Center","site","Site","date","Date","series","Series","title"]:
        if k in df.columns:
            nun=df[k].nunique(dropna=True)
            if 1<nun<=max(20,len(df)//2): return k
    for c in df.columns:
        nun=df[c].nunique(dropna=True)
        if 1<nun<=max(20,len(df)//2): return c
    return None

def infer_batches(meta:pd.DataFrame)->pd.Series:
    for col in ["batch","Batch","run","Run","lane","Lane","plate","Plate","flowcell","Flowcell"]:
        if col in meta.columns: s=meta[col].astype(str); s.index=meta.index; return s
    import re as _re
    def token(s): m=_re.search(r"(FC\w+|L\d{3}|P\d+|\d{4}[-_]\d{2}[-_]\d{2}|\d{8})",s); return m.group(0) if m else "B0"
    return pd.Series([token(x) for x in meta.index.astype(str)], index=meta.index, name="batch")

# ---------- platform + normalization ----------
def detect_type_platform(X:pd.DataFrame):
    vals=X.values.ravel(); vals=vals[np.isfinite(vals)]
    zero=float((X==0).sum().sum())/float(X.size) if X.size else 0.0
    rng=(np.nanpercentile(vals,99.5)-np.nanpercentile(vals,0.5)) if vals.size else 0
    idx=X.index.astype(str)
    has_ilmn=any(str(s).startswith("ILMN_") for s in idx)
    has_affy=any(re.match(r"^\d+_at$",str(s)) for s in idx)
    has_ensg=any(str(s).startswith("ENSG") for s in idx)
    plat="Unknown"
    if has_ilmn or has_affy: plat="Microarray (Illumina/Affy)"
    elif rng>MICROARRAY_RANGE_MAX: plat="Long-read/Counts-like (Illumina/PacBio)"
    elif has_ensg: plat="Short-read RNA-seq (Illumina)"
    return plat,{"zero_fraction":zero,"value_range_approx":rng}

def is_counts_like(X):
    v=X.values.ravel(); v=v[np.isfinite(v)]
    return False if (v.size==0 or np.min(v)<0) else (np.mean(np.isclose(v,np.round(v)))>0.8)

def cpm(C):
    C=C.clip(lower=0).astype(float); lib=C.sum(axis=0).replace(0,np.nan)
    return (C*1e6).div(lib,axis=1).fillna(0.0)

def _tmm_factor(ref,tgt,rl,tl,tm=0.30,ta=0.05):
    g=(ref>0)&(tgt>0); ref,tgt=ref[g],tgt[g]
    if ref.size<5: return 1.0
    M=np.log2((tgt/tl)/(ref/rl)); A=0.5*np.log2((tgt/tl)*(ref/rl)); w=1.0/((tgt/tl)+(ref/rl))
    def keep(x,p):
        if p<=0: return np.ones_like(x,dtype=bool)
        lo,hi=np.quantile(x,p/2),np.quantile(x,1-p/2); return (x>=lo)&(x<=hi)
    m=keep(M,tm)&keep(A,ta)
    if np.sum(m)<5: return 1.0
    return float(2.0**np.average(M[m],weights=w[m]))

def tmm(C:pd.DataFrame)->pd.DataFrame:
    C=C.clip(lower=0).astype(float); libs=C.sum(axis=0).astype(float)
    ref_col=libs.sub(libs.median()).abs().sort_values().index[0]
    ref=C[ref_col].values; rlib=float(libs[ref_col]) or 1.0
    fac={c:(1.0 if c==ref_col else _tmm_factor(ref,C[c].values,rlib,float(libs[c]) or 1.0)) for c in C.columns}
    eff=(libs*pd.Series(fac)).replace(0,np.nan); return (C*1e6).div(eff,axis=1).fillna(0.0)

def choose_norm(X:pd.DataFrame):
    counts=is_counts_like(X)
    lib_cv=float(np.std(X.clip(lower=0).sum(axis=0).values,ddof=1)/max(1e-9,np.mean(X.sum(axis=0).values))) if counts and X.shape[1]>1 else 0.0
    zero=float((X==0).sum().sum())/float(X.size) if X.size else 0.0
    if counts and lib_cv>=0.30: Z=np.log2(tmm(X)+1.0); return Z,"TMM_then_log2",f"Library size CV={lib_cv:.2f} >= 0.30",lib_cv,zero,counts
    if counts and zero>0.25: Z=np.log2(cpm(X)+1.0); return Z,"CPM_then_log2",f"Zero fraction={zero:.2%} > 25%",lib_cv,zero,counts
    if counts: Z=np.log2(X+1.0); return Z,"log2(counts+1)","Counts-like, CV<0.30 and zeros<=25%",lib_cv,zero,counts
    Z=np.log2(X+1.0); return Z,"log2(expr+1)","Non counts-like input",lib_cv,zero,counts

# ---------- harmonize + plots ----------
def _fallback_center(X:pd.DataFrame,batches:pd.Series)->pd.DataFrame:
    Y=X.copy(); gm=Y.values.mean(); gs=Y.values.std(); b=batches.astype(str)
    for k in pd.unique(b):
        cols=b.index[b==k]; m=Y[cols].values.mean(); s=Y[cols].values.std()
        Y[cols]=(Y[cols]-m)*(gs/(s if s>0 else 1))+gm
    return Y

def _fig(path):
    os.makedirs(os.path.dirname(path),exist_ok=True); plt.savefig(path,dpi=300,bbox_inches="tight"); plt.close()

def _pca_np(X, k=2):
    X=X - X.mean(axis=0, keepdims=True); U,S,VT=np.linalg.svd(X, full_matrices=False)
    evr=(S*S)/max(1e-12,(S*S).sum()); scores=U[:,:k]*S[:k] if S.size>=k else U[:,:S.size]*S[:S.size]
    return scores, evr

def _qc_plots(raw,log2,post,meta,figdir):
    os.makedirs(figdir,exist_ok=True)
    # dist
    plt.figure(figsize=(12,6))
    for arr,lbl,a in [(log2.values.ravel(),"Pre",0.5),(post.values.ravel(),"Post",0.5)]:
        arr=arr[np.isfinite(arr)]; plt.hist(arr,bins=120,density=True,alpha=a,label=lbl)
    plt.title("Expression: Pre vs Post (log2)"); plt.legend(); _fig(os.path.join(figdir,"dist_pre_post.png"))
    # corr heatmap
    if post.shape[1]<=600 and post.shape[1]>=2:
        C=np.corrcoef(post.fillna(0).T); plt.figure(figsize=(8,6)); plt.imshow(C,aspect="auto",vmin=-1,vmax=1); plt.colorbar()
        plt.title("Sample correlation (post)"); _fig(os.path.join(figdir,"sample_correlation.png"))
    # PCA (numpy) + scree + scatters
    try:
        Xp=post.T.replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(float)
        if Xp.shape[0]>=2 and Xp.shape[1]>=2:
            scores,evr=_pca_np(Xp,2)
            plt.figure(); plt.plot(range(1,min(11,evr.size)+1), evr[:min(10,evr.size)], marker="o")
            plt.xlabel("PC"); plt.ylabel("Explained variance ratio"); plt.title("PCA Scree (post)")
            _fig(os.path.join(figdir,"pca_scree.png"))
            samples=list(post.columns); M=meta.reindex(samples)
            G=M["group"].astype(str).fillna("NA") if "group" in M.columns else pd.Series(["NA"]*len(samples),index=samples)
            B=M["batch"].astype(str).fillna("NA") if "batch" in M.columns else pd.Series(["NA"]*len(samples),index=samples)
            # group
            plt.figure()
            for g in pd.unique(G): idx=[i for i,s in enumerate(samples) if G.loc[s]==g]; pts=scores[idx,:2]
            plt.scatter(pts[:,0],pts[:,1],alpha=0.7,label=str(g))
            plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA (post) by group"); plt.legend(loc="best",fontsize="small")
            _fig(os.path.join(figdir,"pca_scatter_group.png"))
            # batch
            plt.figure()
            for b in pd.unique(B): idx=[i for i,s in enumerate(samples) if B.loc[s]==b]; pts=scores[idx,:2]
            plt.scatter(pts[:,0],pts[:,1],alpha=0.7,label=str(b))
            plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA (post) by batch"); plt.legend(loc="best",fontsize="x-small",ncol=2)
            _fig(os.path.join(figdir,"pca_scatter_batch.png"))
    except Exception:
        pass
    # library sizes
    try:
        libs=raw.clip(lower=0).sum(axis=0).values; plt.figure(figsize=(12,4)); plt.bar(range(len(libs)),libs); plt.ylabel("Library size"); plt.title("Library size per sample (raw)"); plt.xticks([],[])
        _fig(os.path.join(figdir,"library_sizes.png"))
    except Exception: pass
    # zero fraction
    try:
        zf=((raw==0).sum(axis=0)/max(1,raw.shape[0])).values; plt.figure(figsize=(12,4)); plt.bar(range(len(zf)),zf); plt.ylabel("Zero fraction"); plt.title("Zero fraction per sample (raw)"); plt.xticks([],[])
        _fig(os.path.join(figdir,"zero_fraction_per_sample.png"))
    except Exception: pass

# ---------- main single ----------
def run_single(counts,meta,counts_name=None,meta_name=None,out_root:Optional[str]=None,out_mode:str="co_locate",fig_subdir:str="figs",create_zip:bool=False)->Dict[str,str]:
    if out_mode=="co_locate" and isinstance(counts,(str,os.PathLike)) and os.path.isfile(counts):
        OUTDIR=os.path.join(os.path.dirname(os.path.abspath(counts)),"harmonized")
    else:
        OUTDIR=out_root or os.path.join(os.getcwd(),"out")
    FIG=os.path.join(OUTDIR,fig_subdir); os.makedirs(FIG,exist_ok=True)
    X=read_table(counts,counts_name); meta_base=build_meta_from_cols(list(X.columns),use_prefix=any("__" in c for c in X.columns))
    m=read_meta(meta,meta_name); norm={re.sub(r"[^a-z0-9]","",c.lower()):c for c in m.columns}
    id_col=next((norm[k] for k in ["id","sample","cleanid","bareid"] if k in norm),None)
    if id_col is None:
        cols=set(map(str,meta_base["bare_id"].tolist())); best,ov=None,-1
        for c in m.columns:
            o=len(set(m[c].astype(str).str.strip())&cols)
            if o>ov: ov,best=o,c
        id_col=best
    if id_col is None: raise ValueError("Could not align metadata IDs.")
    m[id_col]=m[id_col].astype(str).str.strip(); m=m.dropna(subset=[id_col]).drop_duplicates(subset=[id_col]); M=m.set_index(id_col)
    meta_df=meta_base.copy(); meta_df["bare_id"]=meta_df["bare_id"].astype(str).str.strip()
    gc=next((c for c in ["group","Group","condition","Condition","phenotype","Phenotype"] if c in M.columns),None)
    if gc is not None: meta_df["group"]=M[gc].reindex(meta_df["bare_id"]).values
    meta_df["group"]=meta_df["group"].fillna("ALL").apply(normalize_group_value)
    bc=_guess_batch_col(m); meta_df["batch"]=M[bc].reindex(meta_df["bare_id"]).values if bc and bc in M.columns else infer_batches(meta_df)
    platform,diags=detect_type_platform(X:=X.replace([np.inf,-np.inf],np.nan))
    log2,norm_name,norm_reason,lib_cv,zero,is_counts=choose_norm(X)
    Xi=log2.apply(lambda r:r.fillna(r.mean()),axis=1).fillna(0); var=Xi.var(axis=1,ddof=1).astype(float).fillna(0.0); Xi=Xi.loc[var>VAR_EPS]
    topk=min(5000,int((var>0).sum())); Xf=Xi.loc[var.nlargest(topk).index] if topk>0 else Xi
    meta_df["batch_collapsed"]=meta_df["batch"].astype(str); batches=meta_df["batch_collapsed"].reindex(Xf.columns).fillna(meta_df["group"].reindex(Xf.columns).astype(str))
    Xh=_fallback_center(Xf,batches); _qc_plots(X,log2,Xh,meta_df,FIG)
    os.makedirs(OUTDIR,exist_ok=True)
    X.to_csv(os.path.join(OUTDIR,"expression_combined.tsv"),sep="\t",encoding="utf-8")
    meta_df.to_csv(os.path.join(OUTDIR,"metadata.tsv"),sep="\t",encoding="utf-8")
    Xh.to_csv(os.path.join(OUTDIR,"expression_harmonized.tsv"),sep="\t",encoding="utf-8")
    pd.DataFrame(index=Xh.columns).to_csv(os.path.join(OUTDIR,"pca_scores.tsv"),sep="\t",encoding="utf-8")
    with open(os.path.join(OUTDIR,"report.json"),"w",encoding="utf-8") as f:
        json.dump({"qc":{"zero_fraction":float(diags["zero_fraction"]),"value_range_approx":float(diags["value_range_approx"]),
                         "harmonization_mode":"fallback_center","platform":platform,"normalization":norm_name,
                         "normalization_reason":norm_reason,"library_size_cv":float(lib_cv),"counts_like":bool(is_counts)},
                   "shapes":{"genes":int(X.shape[0]),"samples":int(X.shape[1])},"notes":{}},f,indent=2)
    with open(os.path.join(OUTDIR,"normalization.txt"),"w",encoding="utf-8") as g:
        g.write(f"normalization: {norm_name}\nreason: {norm_reason}\nlibrary_size_cv: {lib_cv:.6f}\nzero_fraction: {zero:.2%}\ncounts_like: {is_counts}\n")
    zip_path=None
    if create_zip:
        zip_path=os.path.join(OUTDIR,"results_bundle.zip")
        with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as zf:
            for root,_,files in os.walk(OUTDIR):
                for n in files:
                    if n.endswith(".zip"): continue
                    p=os.path.join(root,n); zf.write(p, arcname=os.path.relpath(p, OUTDIR))
    return {"outdir":OUTDIR,"figdir":FIG,"zip":zip_path}

# ---------- multi (co-locate; per-disease archives) ----------
def run_multi(datasets:List[Dict],attempt_combine=True,min_overlap=3000,out_root:Optional[str]=None)->Dict:
    def _safe_name(s:str)->str: return re.sub(r"[^A-Za-z0-9._-]+","_",s or "DS")
    base_out=out_root or os.getcwd(); runs={}; exprs={}; metas={}
    for d in datasets:
        raw=d.get("name","DS"); disease=(d.get("disease") or raw.split("__",1)[0]); dkey=_safe_name(str(disease)); skey=_safe_name(raw)
        res=run_single(d["counts"],d["meta"],d.get("counts_name"),d.get("meta_name"),out_root=None,out_mode="co_locate",create_zip=False)
        res["__disease__"]=dkey; res["__dataset_key__"]=skey; runs[skey]=res
        exprs[skey]=pd.read_csv(os.path.join(res["outdir"],"expression_combined.tsv"),sep="\t",index_col=0,encoding="utf-8")
        metas[skey]=pd.read_csv(os.path.join(res["outdir"],"metadata.tsv"),sep="\t",index_col=0,encoding="utf-8")
    decision={"attempted":attempt_combine,"combined":False,"overlap_genes":0}; combined=None
    if attempt_combine and len(exprs)>=2:
        common=None
        for _,X in exprs.items(): g=set(X.index.astype(str)); common=g if common is None else (common&g)
        decision["overlap_genes"]=len(common or [])
        if len(common or [])>=min_overlap:
            common=list(common); joined=[]; mj=[]
            for n,X in exprs.items(): joined.append(X.loc[common]); M=metas[n].copy(); M["dataset"]=n; mj.append(M)
            expr_all=pd.concat(joined,axis=1); meta_all=pd.concat(mj,axis=0).reset_index().rename(columns={"index":"sample"})
            eb=io.BytesIO(); tmp=expr_all.copy(); tmp.insert(0,"Biomarker",tmp.index); tmp.to_csv(eb,sep="\t",index=False); eb.seek(0)
            mb=io.BytesIO(); meta_all.to_csv(mb,sep="\t",index=False); mb.seek(0)
            combined=run_single(eb,mb,"combined.tsv","combined_meta.tsv",out_root=os.path.join(base_out,"multi_geo_combined"),out_mode="default",create_zip=False)
            decision["combined"]=True
    by_dis={}
    for k,r in runs.items(): by_dis.setdefault(r["__disease__"],[]).append(r)
    disease_archives={}
    for dkey,rs in by_dis.items():
        zpath=os.path.join(base_out,f"harmonizer_{dkey}.zip"); man={"disease":dkey,"datasets":[]}
        with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as zf:
            for r in rs:
                outdir=r["outdir"]; skey=r["__dataset_key__"]
                for root,_,files in os.walk(outdir):
                    for n in files:
                        p=os.path.join(root,n); arc=os.path.join(skey,os.path.relpath(p,outdir)); zf.write(p,arcname=arc)
                man["datasets"].append({"dataset_key":skey,"outdir":outdir,"figdir":r.get("figdir"),"zip":r.get("zip")})
            zf.writestr("manifest.json",json.dumps(man,indent=2))
        disease_archives[dkey]=zpath
    return {"runs":runs,"combine_decision":decision,"combined":combined,"disease_archives":disease_archives}