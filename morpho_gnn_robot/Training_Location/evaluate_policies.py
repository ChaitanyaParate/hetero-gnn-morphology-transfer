import sys, os, json
import numpy as np
import torch
import torch.nn as nn

TRAIN_LOC = '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_Location'
TRAIN_MLP = '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_MLP'
sys.path.insert(0, TRAIN_LOC)

QUAD_URDF = os.path.join(TRAIN_LOC, 'anymal_stripped.urdf')
HEX_URDF  = os.path.join(TRAIN_LOC, 'hexapod_anymal.urdf')
GNN_CKPT  = os.path.join(TRAIN_LOC, 'checkpoints/multi/gnn_ppo_5320704.pt')
MLP_CKPT  = os.path.join(TRAIN_MLP, 'checkpoints/mlp_ppo_10711040.pt')
N_EVAL=20; MAX_STEPS=1000; CMD=np.array([0.7,0.0],dtype=np.float32)

from robot_env_bullet import RobotEnvBullet
from gnn_actor_critic import SlimHeteroGNNActorCritic
from urdf_to_graph    import URDFGraphBuilder

def nf(x,m,v): return np.clip((x-m)/(np.sqrt(v)+1e-8),-10.,10.).astype(np.float32)

def remap_hex(qm,qv):
    hm=np.zeros(42);hv=np.ones(42)
    hm[0:6]=qm[0:6];hv[0:6]=qv[0:6];hm[6:9]=qm[0:3];hv[6:9]=qv[0:3]
    hm[9:15]=qm[6:12];hv[9:15]=qv[6:12];hm[15:18]=qm[6:9];hv[15:18]=qv[6:9]
    hm[18:24]=qm[12:18];hv[18:24]=qv[12:18];hm[24:27]=qm[12:15];hv[24:27]=qv[12:15]
    hm[27:33]=qm[18:24];hv[27:33]=qv[18:24];hm[33:36]=qm[18:21];hv[33:36]=qv[18:21]
    hm[36:42]=qm[24:30];hv[36:42]=qv[24:30]
    return hm,hv

def load_gnn(nj):
    ckpt=torch.load(GNN_CKPT,map_location='cpu',weights_only=False)
    state=ckpt['agent']
    st=state['log_std'].size(0)
    if st<nj:
        exp=torch.full((nj,),state['log_std'].mean().item())
        exp[:st]=state['log_std']; state['log_std']=exp
    m=SlimHeteroGNNActorCritic(node_dim=28,edge_dim=4,hidden_dim=48,num_joints=nj)
    if 'critic_head.0.weight' in state:
        for k in list(state.keys()):
            if k.startswith('critic_head'):
                state.pop(k)
    m.load_state_dict(state, strict=False); m.eval()
    qm=np.array(ckpt.get('obs_norm_mean',np.zeros(30)),dtype=np.float64)
    qv=np.array(ckpt.get('obs_norm_var',np.ones(30)),dtype=np.float64)
    mn,vr=(remap_hex(qm,qv) if nj==18 else (qm,qv))
    return m,mn,vr

def gnn_ep(env,model,builder,nj,mn,vr):
    obs,_=env.reset(); total=0.; fell=True
    for _ in range(MAX_STEPS):
        n2=nj*2; on=obs.copy(); on[:n2]=nf(obs[:n2],mn[:n2],vr[:n2])
        jp=on[:nj].astype(np.float32); jv=on[nj:n2].astype(np.float32)
        blv=on[n2:n2+3].astype(np.float32); bav=on[n2+3:n2+6].astype(np.float32)
        bq=obs[n2+6:n2+10].astype(np.float32); bg=obs[n2+10:n2+13].astype(np.float32)
        g=builder.get_graph(jp,jv,body_quat=bq,body_grav=bg,body_lin_vel=blv,body_ang_vel=bav,command=CMD)
        with torch.no_grad():
            h,_=model._encode(g); jh=model._joint_embeddings(h,g)
            act=model.actor_head(jh).view(1,nj)
        obs,r,term,trunc,info=env.step(act.squeeze(0).numpy()); total+=r
        if term or trunc: fell=info.get('fell',True); break
        fell=False
    return total, not fell

results=[]

print("=== GNN Quadruped ===")
m,mn,vr=load_gnn(12); bq=URDFGraphBuilder(QUAD_URDF,add_body_node=True)
env=RobotEnvBullet(QUAD_URDF,max_episode_steps=MAX_STEPS,render_mode=None)
rw,sc=[],[]
for ep in range(N_EVAL):
    r,s=gnn_ep(env,m,bq,12,mn,vr); rw.append(r); sc.append(int(s))
    print(f"  ep{ep+1}: {r:.1f} {'OK' if s else 'FELL'}")
env.close()
a=np.array(rw)
print(f"  mean={a.mean():.1f}+/-{a.std():.1f} success={np.mean(sc)*100:.0f}%")
results.append(dict(label="GNN – Quadruped (trained)",mean=float(a.mean()),std=float(a.std()),min=float(a.min()),max=float(a.max()),success_rate=float(np.mean(sc)),n=N_EVAL))

print("\n=== GNN Hexapod (zero-shot) ===")
mh,mnh,vrh=load_gnn(18); bh=URDFGraphBuilder(HEX_URDF,add_body_node=True)
env=RobotEnvBullet(HEX_URDF,max_episode_steps=MAX_STEPS,render_mode=None,height_threshold=0.15)
rw,sc=[],[]
for ep in range(N_EVAL):
    r,s=gnn_ep(env,mh,bh,18,mnh,vrh); rw.append(r); sc.append(int(s))
    print(f"  ep{ep+1}: {r:.1f} {'OK' if s else 'FELL'}")
env.close()
a=np.array(rw)
print(f"  mean={a.mean():.1f}+/-{a.std():.1f} success={np.mean(sc)*100:.0f}%")
results.append(dict(label="GNN – Hexapod (zero-shot)",mean=float(a.mean()),std=float(a.std()),min=float(a.min()),max=float(a.max()),success_rate=float(np.mean(sc)),n=N_EVAL))

print("\n=== MLP Quadruped ===")
ckpt_m=torch.load(MLP_CKPT,map_location='cpu',weights_only=False)
state_m=ckpt_m['agent']
mm=np.array(ckpt_m.get('obs_norm_mean',np.zeros(30)),dtype=np.float64)
mv=np.array(ckpt_m.get('obs_norm_var',np.ones(30)),dtype=np.float64)
class MLP(nn.Module):
    def __init__(self,obs,act):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(obs,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh())
        self.actor_head=nn.Sequential(nn.Linear(256,256),nn.Tanh(),nn.Linear(256,act))
        self.log_std=nn.Parameter(torch.zeros(act))
        self.critic_head=nn.Sequential(nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1))
    def _encode(self,x): return self.trunk(x)
mlp=MLP(39,12); mlp.load_state_dict(state_m); mlp.eval()
env=RobotEnvBullet(QUAD_URDF,max_episode_steps=MAX_STEPS,render_mode=None)
rw,sc=[],[]
for ep in range(N_EVAL):
    obs,_=env.reset(); total=0.; fell=True
    for _ in range(MAX_STEPS):
        on=obs.copy(); on[:30]=nf(obs[:30],mm,mv)
        t=torch.tensor(on[:39],dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): act=mlp.actor_head(mlp._encode(t)).squeeze(0).numpy()
        obs,r,term,trunc,info=env.step(act); total+=r
        if term or trunc: fell=info.get('fell',True); break
        fell=False
    rw.append(total); sc.append(int(not fell))
    print(f"  ep{ep+1}: {total:.1f} {'OK' if not fell else 'FELL'}")
env.close()
a=np.array(rw)
print(f"  mean={a.mean():.1f}+/-{a.std():.1f} success={np.mean(sc)*100:.0f}%")
results.append(dict(label="MLP – Quadruped (trained)",mean=float(a.mean()),std=float(a.std()),min=float(a.min()),max=float(a.max()),success_rate=float(np.mean(sc)),n=N_EVAL))

print("\n=== MLP Hexapod (transfer attempt) ===")
mlp_hex=MLP(49,18)
try:
    mlp_hex.load_state_dict(state_m,strict=True); crash="Unexpected success"
except RuntimeError as e:
    crash=str(e)[:400]; print(f"  CRASH: {crash}")
results.append(dict(label="MLP – Hexapod (transfer)",mean=0.,std=0.,min=0.,max=0.,success_rate=0.,n=N_EVAL,note=crash))

out=os.path.join(TRAIN_LOC,'eval_results.json')
with open(out,'w') as f: json.dump(results,f,indent=2)
print("\n\n=== SUMMARY ===")
for r in results:
    print(f"{r['label']:<42} mean={r['mean']:8.1f} +/- {r['std']:6.1f}  success={r['success_rate']*100:5.1f}%")
print(f"Saved: {out}")
