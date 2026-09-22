from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score,roc_curve,roc_auc_score,mean_absolute_error,mean_squared_error,r2_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
BASE=Path(__file__).resolve().parent; PLOTS=BASE/"plots"; MODELS=BASE/"models"; PLOTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
if not (BASE/"titanic.csv").exists(): raise FileNotFoundError("Run 01_eda.py once to create titanic.csv")
df=pd.read_csv(BASE/"titanic.csv"); features=["pclass","sex","age","sibsp","parch","fare","embarked"]; X=df[features]; y=df.survived.astype(int)
print("Class balance\\n",y.value_counts(normalize=True))
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
num=["pclass","age","sibsp","parch","fare"]; cat=["sex","embarked"]
prep=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("enc",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)])
models={"Logistic Regression":LogisticRegression(max_iter=2000,random_state=42),"Decision Tree":DecisionTreeClassifier(max_depth=5,random_state=42),"Random Forest":RandomForestClassifier(n_estimators=200,random_state=42)}
rows=[]; pipes={}
for name,model in models.items():
    pipe=Pipeline([("preprocessor",prep),("model",model)]); pipe.fit(Xtr,ytr); pipes[name]=pipe; pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
    rows.append({"model":name,"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0),"auc":roc_auc_score(yte,prob)})
    fig,ax=plt.subplots(figsize=(5,5)); ConfusionMatrixDisplay(confusion_matrix(yte,pred),display_labels=["Not survived","Survived"]).plot(ax=ax); fig.tight_layout(); fig.savefig(PLOTS/f"{name.lower().replace(' ','_')}_cm.png",dpi=150); plt.close(fig)
res=pd.DataFrame(rows); print("\\nClassification metrics\\n",res); res.to_csv(BASE/"classification_metrics.csv",index=False)
fig,ax=plt.subplots(figsize=(8,6))
for name,p in pipes.items():
    prob=p.predict_proba(Xte)[:,1]; fpr,tpr,_=roc_curve(yte,prob); ax.plot(fpr,tpr,label=f"{name} AUC={roc_auc_score(yte,prob):.3f}")
ax.plot([0,1],[0,1],"--"); ax.legend(); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); fig.tight_layout(); fig.savefig(PLOTS/"roc_curves.png",dpi=150); plt.close(fig)
tree=pipes["Decision Tree"]; names=tree.named_steps["preprocessor"].get_feature_names_out(); fig,ax=plt.subplots(figsize=(22,12)); plot_tree(tree.named_steps["model"],feature_names=names,class_names=["Not survived","Survived"],ax=ax); fig.tight_layout(); fig.savefig(PLOTS/"decision_tree.png",dpi=150); plt.close(fig)
imb=[]
for label,model in [("baseline",LogisticRegression(max_iter=2000,random_state=42)),("class_weight_balanced",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42))]:
    p=Pipeline([("preprocessor",prep),("model",model)]); p.fit(Xtr,ytr); pred=p.predict(Xte); imb.append({"variant":label,"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0)})
sm=ImbPipeline([("preprocessor",prep),("smote",SMOTE(random_state=42)),("model",LogisticRegression(max_iter=2000,random_state=42))]); sm.fit(Xtr,ytr); pred=sm.predict(Xte); imb.append({"variant":"SMOTE_training_only","precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0)})
imb=pd.DataFrame(imb); print("\\nImbalance comparison\\n",imb); imb.to_csv(BASE/"imbalance_metrics.csv",index=False)
rf=Pipeline([("preprocessor",prep),("model",RandomForestClassifier(oob_score=True,random_state=42,n_jobs=-1))]); grid=GridSearchCV(rf,{"model__n_estimators":[100,200],"model__max_depth":[None,5,10],"model__max_features":["sqrt","log2"]},scoring="f1",cv=5,n_jobs=-1); grid.fit(Xtr,ytr); best_rf=grid.best_estimator_; print("\\nBest params",grid.best_params_); print("CV F1",grid.best_score_); print("OOB",best_rf.named_steps["model"].oob_score_); (BASE/"grid_search_report.md").write_text(f"# Grid Search\\nBest parameters: `{grid.best_params_}`\\n\\nCV F1: `{grid.best_score_:.6f}`\\n\\nOOB score: `{best_rf.named_steps['model'].oob_score_:.6f}`\\n",encoding="utf-8")
rfpred=best_rf.predict(Xte); rfprob=best_rf.predict_proba(Xte)[:,1]; tuned={"model":"Random Forest (tuned)","accuracy":accuracy_score(yte,rfpred),"precision":precision_score(yte,rfpred,zero_division=0),"recall":recall_score(yte,rfpred,zero_division=0),"f1":f1_score(yte,rfpred,zero_division=0),"auc":roc_auc_score(yte,rfprob)}; comparison=pd.concat([res,pd.DataFrame([tuned])],ignore_index=True)
reg_features=[c for c in ["pclass","sex","age","sibsp","parch","embarked"] if c in df.columns]; xr=df[reg_features]; yr=df.fare; xrtr,xrte,yrtr,yrte=train_test_split(xr,yr,test_size=.2,random_state=42); rnum=[c for c in reg_features if pd.api.types.is_numeric_dtype(xr[c])]; rcat=[c for c in reg_features if c not in rnum]; rprep=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),rnum),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("enc",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),rcat)]); reg=Pipeline([("preprocessor",rprep),("model",LinearRegression())]); reg.fit(xrtr,yrtr); yp=reg.predict(xrte); mae=mean_absolute_error(yrte,yp); rmse=np.sqrt(mean_squared_error(yrte,yp)); r2=r2_score(yrte,yp); p=reg.named_steps["preprocessor"].transform(xrte).shape[1]; n=len(yrte); adj=1-(1-r2)*(n-1)/(n-p-1) if n-p-1>0 else np.nan; pd.DataFrame([{"MAE":mae,"RMSE":rmse,"R2":r2,"Adjusted_R2":adj}]).to_csv(BASE/"regression_metrics.csv",index=False); resid=yrte-yp; fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(yp,resid,alpha=.65); ax.axhline(0,linestyle="--"); ax.set_xlabel("Predicted fare"); ax.set_ylabel("Residual"); fig.tight_layout(); fig.savefig(PLOTS/"fare_residuals.png",dpi=150); plt.close(fig)
comparison.to_csv(BASE/"model_comparison.csv",index=False); best=comparison.sort_values(["f1","auc"],ascending=False).iloc[0]; full=best_rf if best.model=="Random Forest (tuned)" else pipes[best.model]; joblib.dump(full,MODELS/"best_pipeline.joblib"); loaded=joblib.load(MODELS/"best_pipeline.joblib"); print("Reloaded raw prediction:",loaded.predict(Xte.iloc[[0]])); print("\\nSelected classifier:",best.model); print("Regression MAE/RMSE/R2/AdjR2:",mae,rmse,r2,adj)
