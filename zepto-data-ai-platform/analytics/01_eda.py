from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent; PLOTS=BASE/"plots"; PLOTS.mkdir(exist_ok=True); CSV=BASE/"titanic.csv"
if CSV.exists(): df=pd.read_csv(CSV)
else:
    df=sns.load_dataset("titanic")
    df.to_csv(CSV,index=False)
print("\\nINFO"); df.info(); print("\\nDESCRIBE\\n",df.describe(include="all")); print("\\nSHAPE",df.shape)
missing=(df.isna().mean()*100).loc[lambda s:s>0].sort_values(ascending=False); print("\\nMISSING %\\n",missing)
clean=df.copy()
for col,pct in missing.items():
    if pct<5: clean=clean.dropna(subset=[col])
    elif pct<=30:
        if pd.api.types.is_numeric_dtype(clean[col]): clean[col]=clean[col].fillna(clean[col].median())
        else: clean[col]=clean[col].fillna(clean[col].mode()[0])
    else:
        if clean[col].dtype=="object" or str(clean[col].dtype)=="category": clean[col]=clean[col].astype(object).fillna("Missing")
        else: clean=clean.drop(columns=[col])
clean.to_csv(BASE/"cleaned_titanic.csv",index=False)

def iqr(s):
    q1,q3=s.quantile([.25,.75]); i=q3-q1; return int(((s<q1-1.5*i)|(s>q3+1.5*i)).sum()),q1-1.5*i,q3+1.5*i
for col in ["age","fare"]:
    n,lo,hi=iqr(clean[col].dropna()); print(f"{col} outliers={n}, bounds=({lo:.3f},{hi:.3f})")
    fig,ax=plt.subplots(figsize=(7,4)); sns.histplot(clean[col].dropna(),kde=True,ax=ax); fig.tight_layout(); fig.savefig(PLOTS/f"{col}_hist.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3)); sns.boxplot(x=clean[col],ax=ax); fig.tight_layout(); fig.savefig(PLOTS/f"{col}_box.png",dpi=150); plt.close(fig)
mean,median,mode=clean.fare.mean(),clean.fare.median(),clean.fare.mode().iloc[0]
skew="right-skewed" if mean>median>mode else "left-skewed" if mean<median<mode else "not strictly determined by ordering"
print("Fare mean/median/mode:",mean,median,mode,"=>",skew)
print("\\nSurvival by sex\\n",clean.groupby("sex").survived.mean())
print("\\nSurvival by pclass\\n",clean.groupby("pclass").survived.mean())
print("\\nSurvival by sex+pclass\\n",clean.groupby(["sex","pclass"]).survived.mean())
cols=["survived","pclass","age","sibsp","parch","fare"]; corr=clean[cols].corr(); print("\\nCorrelation\\n",corr)
fig,ax=plt.subplots(figsize=(8,6)); sns.heatmap(corr,annot=True,fmt=".2f",ax=ax); fig.tight_layout(); fig.savefig(PLOTS/"correlation_heatmap.png",dpi=150); plt.close(fig)
pairs=[]
for i,a in enumerate(cols):
    for b in cols[i+1:]: pairs.append((a,b,corr.loc[a,b],abs(corr.loc[a,b])))
pairs.sort(key=lambda x:x[3],reverse=True); print("Top two correlations:",pairs[:2])
fig,ax=plt.subplots(figsize=(8,5)); sns.barplot(data=clean,x="pclass",y="survived",hue="sex",errorbar=None,ax=ax); fig.tight_layout(); fig.savefig(PLOTS/"survival_class_sex.png",dpi=150); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5)); sns.boxplot(data=clean,x="survived",y="fare",hue="pclass",ax=ax); fig.tight_layout(); fig.savefig(PLOTS/"fare_survival_class.png",dpi=150); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5)); sns.scatterplot(data=clean,x="age",y="fare",hue="survived",style="pclass",alpha=.65,ax=ax); fig.tight_layout(); fig.savefig(PLOTS/"age_fare_survival.png",dpi=150); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5)); sns.barplot(data=clean,x="embarked",y="survived",hue="sex",errorbar=None,ax=ax); fig.tight_layout(); fig.savefig(PLOTS/"survival_embarked_sex.png",dpi=150); plt.close(fig)
for col in ["age","fare"]:
    before_mean=clean[col].mean(); before_std=clean[col].std()
    z=(clean[col]-before_mean)/before_std
    print(f"{col} BEFORE mean/std:",before_mean,before_std)
    print(f"{col} AFTER standardized mean/std:",z.mean(),z.std())
report=f"""# EDA Report\n\nMissing percentages:\n\n{missing.to_string()}\n\nFare mean={mean:.4f}, median={median:.4f}, mode={mode:.4f}; distribution={skew}.\n\nStrongest absolute correlations:\n1. {pairs[0][0]} vs {pairs[0][1]} = {pairs[0][2]:.4f}\n2. {pairs[1][0]} vs {pairs[1][1]} = {pairs[1][2]:.4f}\n\n## Chart interpretations\n### Survival by class and sex\nThe chart jointly compares class and sex, showing how survival varies across both dimensions. This is more informative than examining either variable independently.\n\n### Fare by survival and class\nThe chart compares fare distributions across survival outcomes while retaining passenger class. It shows the association between ticket price/class and survival.\n\n### Age vs fare\nThis chart combines age, fare, survival and class to show where different outcome groups occur in feature space.\n\n### Embarkation and sex\nThis chart compares survival rates across embarkation points while preserving sex as a second grouping variable.\n"""
(BASE/"eda_report.md").write_text(report,encoding="utf-8")
print("EDA complete. Commit titanic.csv after first online run.")
