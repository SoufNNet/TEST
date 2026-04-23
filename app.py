"""
Streamlit app — Prédiction ΔH°f et S° à partir de SMILES
Lancer avec : streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import re
import io
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# ── Tentative d'import RDKit ────────────────────────────────────────────────
try:
 from rdkit import Chem
 from rdkit.Chem import Descriptors, Draw
 from rdkit.ML.Descriptors import MoleculeDescriptors
 RDKIT_OK = True
except ImportError:
 RDKIT_OK = False

# ── Constantes ───────────────────────────────────────────────────────────────
R_GAS_CONSTANT = 8.314
TEMPERATURE = 298.15
ADIM_TO_KJMOL = R_GAS_CONSTANT * TEMPERATURE / 1000.0

# ════════════════════════════════════════════════════════════════════════════
# Modèles PyTorch (copie exacte de votre code)
# ════════════════════════════════════════════════════════════════════════════

def init_params(m):
 for _, module in m.named_modules():
  for param_name, param in module.named_parameters():
   if 'weight' in param_name:
    if any(k in param_name for k in ('conv', 'lin', 'ih')):
     nn.init.xavier_uniform_(param)
    elif 'hh' in param_name:
     nn.init.orthogonal_(param)
   elif param_name == 'bias':
    nn.init.constant_(param, 0.0)


class Vocab:
 def __init__(self, tokens):
  self.itos = tokens
  self.stoi = {t: i for i, t in enumerate(tokens)}
 def __len__(self):
  return len(self.itos)


class Encoder(nn.Module):
 def __init__(self, input_size=48, hidden_size=512, n_layers=2,
     bidirectional=True, latent_size=56):
  super().__init__()
  self.hidden_factor = (2 if bidirectional else 1) * n_layers
  self.rnn = nn.GRU(input_size, hidden_size, n_layers,
       bidirectional=bidirectional, batch_first=True)
  self.mean_lin = nn.Linear(hidden_size * self.hidden_factor, latent_size)
  self.logvar_lin = nn.Linear(hidden_size * self.hidden_factor, latent_size)
  init_params(self)

 def forward(self, x):
  _, h = self.rnn(x)
  h = h.permute(1, 0, 2).contiguous().view(h.size(1), -1)
  return self.mean_lin(h), -torch.abs(self.logvar_lin(h))


class Decoder(nn.Module):
 def __init__(self, input_size=48, hidden_size=512, n_layers=4,
     dropout=0.5, latent_size=56, vocab_size=64,
     max_len=75, vocab=None, sos_idx=2, padding_idx=1):
  super().__init__()
  self.hidden_size = hidden_size
  self.hidden_factor = n_layers
  self.embedding_dropout = nn.Dropout(dropout)
  self.rnn    = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
  self.latent2hidden  = nn.Linear(latent_size, hidden_size * n_layers)
  self.outputs2vocab  = nn.Linear(hidden_size, vocab_size)
  self.outputs_dropout = nn.Dropout(dropout)
  init_params(self)

 def forward(self, emb, z):
  h = self.latent2hidden(z)
  h = torch.tanh(h.view(-1, self.hidden_factor,
        self.hidden_size).permute(1, 0, 2).contiguous())
  emb = self.embedding_dropout(emb)
  out, _ = self.rnn(emb, h)
  b, s, hs = out.size()
  return self.outputs2vocab(self.outputs_dropout(out.view(-1, hs))).view(b, s, -1)


class Vae(nn.Module):
 def __init__(self, vocab, vocab_size, embedding_size, dropout, padding_idx,
     sos_idx, unk_idx, max_len, n_layers, hidden_size,
     bidirectional=True, latent_size=56, partialsmiles=False):
  super().__init__()
  self.vocab  = vocab
  self.embedding = nn.Embedding(vocab_size, embedding_size)
  self.encoder = Encoder(embedding_size, hidden_size, n_layers,
         bidirectional, latent_size)
  dec_layers  = (2 if bidirectional else 1) * n_layers
  self.decoder = Decoder(embedding_size, hidden_size, dec_layers,
         dropout, latent_size, vocab_size,
         max_len, vocab, sos_idx, padding_idx)

 def encode_to_mean(self, x):
  x = x.cuda() if next(self.parameters()).is_cuda else x
  emb = self.embedding(x)
  mean, _ = self.encoder(emb)
  return mean


class ANNRegressor(nn.Module):
 def __init__(self, input_size, hidden_layers, dropout_rate=0.3):
  super().__init__()
  layers, prev = [], input_size
  for hs in hidden_layers:
   layers += [nn.Linear(prev, hs), nn.ReLU(),
      nn.BatchNorm1d(hs), nn.Dropout(dropout_rate)]
   prev = hs
  layers.append(nn.Linear(prev, 1))
  self.network = nn.Sequential(*layers)

 def forward(self, x):
  return self.network(x).squeeze()


class DescriptorProcessor:
 def __init__(self):
  self.columns_after_dropna = None
  self.variance_selector  = None
  self.final_descriptor_names = None
  self.scaler_desc   = None

 def transform(self, desc_arr, desc_names):
  df  = pd.DataFrame(desc_arr, columns=desc_names)
  df  = df[self.columns_after_dropna]
  df_var = pd.DataFrame(
   self.variance_selector.transform(df),
   columns=df.columns[self.variance_selector.get_support()])
  return self.scaler_desc.transform(df_var[self.final_descriptor_names])


# ════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ════════════════════════════════════════════════════════════════════════════

def tokenize_smiles(smiles):
 pattern = r'\[[^\]]+\]|%\d{2}|Br|Cl|se|as|@@|[BCNOPSFIbcnops]|[=#\-+:\/\\().\[\]@]|\d'
 return re.findall(pattern, str(smiles))


def smiles_to_tensor(smiles_list, vocab, max_len=75):
 data = []
 for smi in smiles_list:
  toks = tokenize_smiles(smi)
  idx = ([vocab.stoi['<sos>']] +
    [vocab.stoi.get(t, vocab.stoi['<unk>']) for t in toks])[:max_len]
  idx += [vocab.stoi[' ']] * (max_len - len(idx))
  data.append(torch.LongTensor(idx))
 return torch.stack(data)


def extract_latent(vae, smiles_list, vocab, device, batch=128):
 vae.eval()
 vecs = []
 with torch.no_grad():
  for i in range(0, len(smiles_list), batch):
   x = smiles_to_tensor(smiles_list[i:i+batch], vocab).to(device)
   vecs.append(vae.encode_to_mean(x).cpu().numpy())
 return np.vstack(vecs)


def canonicalize(smi):
 if not RDKIT_OK:
  return smi
 mol = Chem.MolFromSmiles(str(smi))
 return Chem.MolToSmiles(mol, canonical=True) if mol else None


def calc_descriptors(smiles_list, desc_names):
 if not RDKIT_OK:
  return [], [], []
 calc  = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
 rows, vi, errs = [], [], []
 for i, smi in enumerate(smiles_list):
  try:
   mol = Chem.MolFromSmiles(smi)
   if mol:
    d = list(calc.CalcDescriptors(mol))
    if all(np.isfinite(v) for v in d):
     rows.append(d); vi.append(i)
    else:
     errs.append(smi)
   else:
    errs.append(smi)
  except Exception:
   errs.append(smi)
 return rows, vi, errs


def mol_to_image(smi):
 """Renvoie une image PIL de la molécule."""
 if not RDKIT_OK:
  return None
 try:
  mol = Chem.MolFromSmiles(smi)
  if mol:
   return Draw.MolToImage(mol, size=(220, 160))
 except Exception:
  pass
 return None


# ════════════════════════════════════════════════════════════════════════════
# Chargement des pipelines (mis en cache)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_pipeline(sklearn_path, ann_path, vae_cfg_path, vae_weights_path):
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 with open(sklearn_path, "rb") as f:
  bundle = pickle.load(f)
 scaler  = bundle["scaler"]
 dp   = bundle["descriptor_processor"]
 desc_names = bundle["safe_descriptor_names"]
 hidden  = bundle["hidden_layers"]

 with open(vae_cfg_path, "rb") as f:
  vb = pickle.load(f)
 vocab = vb["vocab"]
 vcfg = vb["vae_config"]

 ann_ckpt = torch.load(ann_path, map_location=device, weights_only=False)
 input_size = scaler.mean_.shape[0]
 ann  = ANNRegressor(input_size, hidden).to(device)
 ann.load_state_dict({k: v for k, v in ann_ckpt["model_state_dict"].items()
       if not k.endswith((".x", ".y"))}, strict=True)
 ann.eval()

 vae = Vae(
  vocab=vocab, vocab_size=vcfg["vocab_size"],
  embedding_size=vcfg["embedding_size"], dropout=vcfg["dropout"],
  padding_idx=vcfg["padding_idx"], sos_idx=vcfg["sos_idx"],
  unk_idx=vcfg["unk_idx"], max_len=vcfg["max_len"],
  n_layers=vcfg["n_layers"], hidden_size=vcfg["hidden_size"],
  bidirectional=True, latent_size=vcfg["latent_size"],
 )
 vae_ckpt = torch.load(vae_weights_path, map_location=device, weights_only=False)
 vae.load_state_dict(vae_ckpt["model_state_dict"])
 vae = vae.to(device)
 vae.eval()

 return ann, scaler, dp, vae, vocab, desc_names, device


def predict_smiles(smiles_list, ann, scaler, dp, vae, vocab, desc_names, device, conv):
 canonical = []
 for smi in smiles_list:
  c = canonicalize(smi.strip())
  canonical.append(c)

 raw, vi, errs = calc_descriptors(
  [c for c in canonical if c], desc_names)

 if not vi:
  return pd.DataFrame(), errs

 valid_smi = [c for c in canonical if c]
 valid_smi_f = [valid_smi[i] for i in vi]
 desc_arr = np.array(raw)
 desc_norm = dp.transform(desc_arr, desc_names)
 lat   = extract_latent(vae, valid_smi_f, vocab, device)
 X   = scaler.transform(np.concatenate([lat, desc_norm], axis=1))

 ann.eval()
 with torch.no_grad():
  y = ann(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy() * conv

 df = pd.DataFrame({
  "SMILES (entré)" : [smiles_list[i] for i in vi],
  "SMILES canonique" : valid_smi_f,
  "Valeur prédite" : np.round(y, 2),
 })
 return df, errs


# ════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ════════════════════════════════════════════════════════════════════════════

try:
 from streamlit_ketcher import st_ketcher
 KETCHER_OK = True
except ImportError:
 KETCHER_OK = False

st.set_page_config(
 page_title="ΔH°f / S° Prediction",
 page_icon=None,
 layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
.page-title {
 font-size: 22px;
 font-weight: 700;
 line-height: 1.35;
 margin-bottom: 4px;
 color: var(--text-color);
}
.page-sub {
 font-size: 13px;
 color: #888;
 margin-bottom: 0;
}
.result-card {
 background: #f4f6fb;
 border-radius: 12px;
 padding: 22px 28px;
 border-left: 5px solid #4a90d9;
 margin: 8px 0;
}
.result-card .label { font-size: 13px; color: #666; margin: 0 0 4px 0; }
.result-card .value { font-size: 28px; font-weight: 700; color: #1a1a2e; margin: 0; }
.result-card .unit { font-size: 13px; color: #888; }
.warn-box {
 background: #fff3cd;
 border-left: 4px solid #ffc107;
 border-radius: 6px;
 padding: 10px 14px;
 font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(
 '<p class="page-title">Hybrid Molecular Representation Combining Variational '
 'Autoencoder Latent Space and Physicochemical Descriptors for the Prediction '
 'of Standard Enthalpy of Formation and Standard Entropy</p>',
 unsafe_allow_html=True,
)
st.markdown(
 '<p class="page-sub">Standard enthalpy of formation <b>ΔH°f</b> (kJ/mol) · '
 'Standard entropy <b>S°</b> (J/mol·K) · T = 298.15 K</p>',
 unsafe_allow_html=True,
)
st.markdown("---")

# ── Model paths (hardcoded — files committed to GitHub repo) ──────────────────
H_SKLEARN = "pipeline_enthalpy_saved/pipeline_enthalpy_sklearn.pkl"
H_ANN  = "pipeline_enthalpy_saved/pipeline_enthalpy_ann.pt"
H_CFG  = "pipeline_enthalpy_saved/pipeline_enthalpy_vae_config.pkl"
S_SKLEARN = "pipeline_entropy_saved/pipeline_entropy_sklearn.pkl"
S_ANN  = "pipeline_entropy_saved/pipeline_entropy_ann.pt"
S_CFG  = "pipeline_entropy_saved/pipeline_entropy_vae_config.pkl"
VAE_W  = "vae_molecule_best_van_rachid_poison_VAE_new_128.pt"

# ── Auto-load models at startup ───────────────────────────────────────────────
if "pipelines" not in st.session_state:
 with st.spinner("Loading models, please wait..."):
  try:
   ann_H, sc_H, dp_H, vae_H, vocab_H, dn_H, dev = load_pipeline(
    H_SKLEARN, H_ANN, H_CFG, VAE_W)
   ann_S, sc_S, dp_S, vae_S, vocab_S, dn_S, _ = load_pipeline(
    S_SKLEARN, S_ANN, S_CFG, VAE_W)
   st.session_state["pipelines"] = {
    "H": (ann_H, sc_H, dp_H, vae_H, vocab_H, dn_H, dev),
    "S": (ann_S, sc_S, dp_S, vae_S, vocab_S, dn_S, dev),
   }
  except Exception as e:
   st.error(f"Model loading error: {e}")

models_loaded = "pipelines" in st.session_state

# ── SMILES input — two modes ──────────────────────────────────────────────────
st.subheader("Molecule Input")

if "smiles_input" not in st.session_state:
 st.session_state["smiles_input"] = ""

mode = st.radio(
 "Input method",
 ["Type SMILES", "Draw molecule"],
 horizontal=True,
 label_visibility="collapsed",
)

smiles_to_predict = ""

if mode == "Type SMILES":
 smiles_to_predict = st.text_input(
  "Enter a SMILES string",
  placeholder="e.g. CCO or c1ccccc1 or CC(=O)O",
  help="Standard SMILES notation for a single molecule.",
  key="smiles_text",
 ).strip()

else:
 if not KETCHER_OK:
  st.warning(
   "The molecule editor requires `streamlit-ketcher`. "
   "Add it to requirements.txt and redeploy."
  )
 else:
  st.caption("Draw your molecule below, then click **Apply** in the editor.")
  ketcher_smi = st_ketcher(st.session_state.get("smiles_input", ""), height=420)
  if ketcher_smi:
   smiles_to_predict = ketcher_smi.strip()
   st.session_state["smiles_input"] = smiles_to_predict

if smiles_to_predict:
 canon = canonicalize(smiles_to_predict)
 if canon:
  st.caption(f"Canonical SMILES: `{canon}`")
 else:
  st.warning("Could not parse SMILES — please check the input.")
  smiles_to_predict = ""

st.markdown("")
predict_btn = st.button("Predict", type="primary")

# ════════════════════════════════════════════════════════════════════════════
# Prediction
# ════════════════════════════════════════════════════════════════════════════

if predict_btn:
 if not smiles_to_predict:
  st.warning("Please enter or draw a molecule first.")
 elif not models_loaded:
  st.error("Models are not loaded. Check that all model files are present in the repository.")
 else:
  pipes = st.session_state["pipelines"]
  with st.spinner("Computing prediction…"):
   df_H, errs_H = predict_smiles([smiles_to_predict], *pipes["H"], conv=ADIM_TO_KJMOL)
   df_S, errs_S = predict_smiles([smiles_to_predict], *pipes["S"], conv=R_GAS_CONSTANT)

  if errs_H or errs_S:
   st.markdown(
    '<div class="warn-box">Could not compute descriptors for this molecule. '
    'Try a different SMILES.</div>', unsafe_allow_html=True)
  elif df_H.empty and df_S.empty:
   st.error("No result obtained. Please verify the SMILES.")
  else:
   st.markdown("---")
   st.subheader("Prediction Results")

   smi_can = canonicalize(smiles_to_predict)
   c_img, c_h, c_s = st.columns([1.2, 1, 1])

   with c_img:
    img = mol_to_image(smi_can)
    if img:
     st.image(img, caption=smi_can, use_container_width=True)

   with c_h:
    if not df_H.empty:
     val_H = df_H["Valeur prédite"].iloc[0]
     st.markdown(
      f'<div class="result-card">'
      f'<p class="label">Standard Enthalpy of Formation</p>'
      f'<p class="value">{val_H:.2f}</p>'
      f'<p class="unit">kJ/mol &nbsp;·&nbsp; ΔH°f</p>'
      f'</div>', unsafe_allow_html=True)

   with c_s:
    if not df_S.empty:
     val_S = df_S["Valeur prédite"].iloc[0]
     st.markdown(
      f'<div class="result-card">'
      f'<p class="label">Standard Entropy</p>'
      f'<p class="value">{val_S:.2f}</p>'
      f'<p class="unit">J/mol·K &nbsp;·&nbsp; S°</p>'
      f'</div>', unsafe_allow_html=True)

   # Download
   result_df = pd.DataFrame({
    "SMILES (input)" : [smiles_to_predict],
    "Canonical SMILES" : [smi_can],
    "ΔH°f (kJ/mol)" : [round(val_H, 2) if not df_H.empty else None],
    "S° (J/mol·K)"  : [round(val_S, 2) if not df_S.empty else None],
   })
   buf = io.BytesIO()
   result_df.to_excel(buf, index=False, engine="openpyxl")
   st.download_button(
    label=" Download result (Excel)",
    data=buf.getvalue(),
    file_name="thermodynamic_prediction.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
   )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
 "Model: VAE latent space + RDKit physicochemical descriptors + ANN regression • "
 "T = 298.15 K • R = 8.314 J/mol·K"
)
