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
TEMPERATURE    = 298.15
ADIM_TO_KJMOL  = R_GAS_CONSTANT * TEMPERATURE / 1000.0

# ════════════════════════════════════════════════════════════════════════════
#  Modèles PyTorch (copie exacte de votre code)
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
        self.mean_lin   = nn.Linear(hidden_size * self.hidden_factor, latent_size)
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
        self.hidden_size   = hidden_size
        self.hidden_factor = n_layers
        self.emb_drop      = nn.Dropout(dropout)
        self.rnn           = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
        self.lat2hid       = nn.Linear(latent_size, hidden_size * n_layers)
        self.out2vocab     = nn.Linear(hidden_size, vocab_size)
        self.out_drop      = nn.Dropout(dropout)
        init_params(self)

    def forward(self, emb, z):
        h = self.lat2hid(z)
        h = torch.tanh(h.view(-1, self.hidden_factor,
                               self.hidden_size).permute(1, 0, 2).contiguous())
        emb = self.emb_drop(emb)
        out, _ = self.rnn(emb, h)
        b, s, hs = out.size()
        return self.out2vocab(self.out_drop(out.view(-1, hs))).view(b, s, -1)


class Vae(nn.Module):
    def __init__(self, vocab, vocab_size, embedding_size, dropout, padding_idx,
                 sos_idx, unk_idx, max_len, n_layers, hidden_size,
                 bidirectional=True, latent_size=56, partialsmiles=False):
        super().__init__()
        self.vocab     = vocab
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder   = Encoder(embedding_size, hidden_size, n_layers,
                                  bidirectional, latent_size)
        dec_layers     = (2 if bidirectional else 1) * n_layers
        self.decoder   = Decoder(embedding_size, hidden_size, dec_layers,
                                  dropout, latent_size, vocab_size,
                                  max_len, vocab, sos_idx, padding_idx)

    def encode_to_mean(self, x):
        x   = x.cuda() if next(self.parameters()).is_cuda else x
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
        self.columns_after_dropna   = None
        self.variance_selector      = None
        self.final_descriptor_names = None
        self.scaler_desc            = None

    def transform(self, desc_arr, desc_names):
        df     = pd.DataFrame(desc_arr, columns=desc_names)
        df     = df[self.columns_after_dropna]
        df_var = pd.DataFrame(
            self.variance_selector.transform(df),
            columns=df.columns[self.variance_selector.get_support()])
        return self.scaler_desc.transform(df_var[self.final_descriptor_names])


# ════════════════════════════════════════════════════════════════════════════
#  Fonctions utilitaires
# ════════════════════════════════════════════════════════════════════════════

def tokenize_smiles(smiles):
    pattern = r'\[[^\]]+\]|%\d{2}|Br|Cl|se|as|@@|[BCNOPSFIbcnops]|[=#\-+:\/\\().\[\]@]|\d'
    return re.findall(pattern, str(smiles))


def smiles_to_tensor(smiles_list, vocab, max_len=75):
    data = []
    for smi in smiles_list:
        toks = tokenize_smiles(smi)
        idx  = ([vocab.stoi['<sos>']] +
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
    calc     = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
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
#  Chargement des pipelines (mis en cache)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_pipeline(sklearn_path, ann_path, vae_cfg_path, vae_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(sklearn_path, "rb") as f:
        bundle = pickle.load(f)
    scaler     = bundle["scaler"]
    dp         = bundle["descriptor_processor"]
    desc_names = bundle["safe_descriptor_names"]
    hidden     = bundle["hidden_layers"]

    with open(vae_cfg_path, "rb") as f:
        vb = pickle.load(f)
    vocab  = vb["vocab"]
    vcfg   = vb["vae_config"]

    ann_ckpt   = torch.load(ann_path, map_location=device, weights_only=False)
    input_size = scaler.mean_.shape[0]
    ann        = ANNRegressor(input_size, hidden).to(device)
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

    valid_smi   = [c for c in canonical if c]
    valid_smi_f = [valid_smi[i] for i in vi]
    desc_arr    = np.array(raw)
    desc_norm   = dp.transform(desc_arr, desc_names)
    lat         = extract_latent(vae, valid_smi_f, vocab, device)
    X           = scaler.transform(np.concatenate([lat, desc_norm], axis=1))

    ann.eval()
    with torch.no_grad():
        y = ann(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy() * conv

    df = pd.DataFrame({
        "SMILES (entré)"   : [smiles_list[i] for i in vi],
        "SMILES canonique" : valid_smi_f,
        "Valeur prédite"   : np.round(y, 2),
    })
    return df, errs


# ════════════════════════════════════════════════════════════════════════════
#  Interface Streamlit
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Prédiction ΔH°f / S°",
    page_icon="⚗️",
    layout="wide",
)

# ── CSS minimal ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #4a90d9;
    margin-bottom: 10px;
}
.metric-card h4 { margin: 0 0 4px 0; font-size: 13px; color: #666; }
.metric-card p  { margin: 0; font-size: 22px; font-weight: 600; color: #1a1a2e; }
.warn-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 13px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Titre ────────────────────────────────────────────────────────────────────
st.title("⚗️  Prédiction de propriétés thermodynamiques")
st.caption("Enthalpy de formation standard **ΔH°f** (kJ/mol) · Entropie standard **S°** (J/mol·K)")

if not RDKIT_OK:
    st.error("⚠️  **RDKit non installé.** Installez-le avec `pip install rdkit` puis redémarrez l'app.")

# ════════════════════════════════════════════════════════════════════════════
#  Sidebar — chemins des fichiers
# ════════════════════════════════════════════════════════════════════════════

# ── Chemins des modèles (hardcodés — fichiers dans le repo GitHub) ────────────
H_SKLEARN = "pipeline_enthalpy_saved/pipeline_enthalpy_sklearn.pkl"
H_ANN     = "pipeline_enthalpy_saved/pipeline_enthalpy_ann.pt"
H_CFG     = "pipeline_enthalpy_saved/pipeline_enthalpy_vae_config.pkl"
S_SKLEARN = "pipeline_entropy_saved/pipeline_entropy_sklearn.pkl"
S_ANN     = "pipeline_entropy_saved/pipeline_entropy_ann.pt"
S_CFG     = "pipeline_entropy_saved/pipeline_entropy_vae_config.pkl"
VAE_W     = "vae_molecule_best_van_rachid_poison_VAE_new_128.pt"

# ── Chargement automatique au démarrage ──────────────────────────────────────
models_loaded = False

if "pipelines" not in st.session_state:
    with st.spinner("⏳ Chargement des modèles en cours…"):
        try:
            ann_H, sc_H, dp_H, vae_H, vocab_H, dn_H, dev = load_pipeline(
                H_SKLEARN, H_ANN, H_CFG, VAE_W)
            ann_S, sc_S, dp_S, vae_S, vocab_S, dn_S, _   = load_pipeline(
                S_SKLEARN, S_ANN, S_CFG, VAE_W)
            st.session_state["pipelines"] = {
                "H": (ann_H, sc_H, dp_H, vae_H, vocab_H, dn_H, dev),
                "S": (ann_S, sc_S, dp_S, vae_S, vocab_S, dn_S, dev),
            }
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des modèles : {e}")

models_loaded = "pipelines" in st.session_state

with st.sidebar:
    if models_loaded:
        st.success("✅  Modèles chargés")
    else:
        st.error("❌  Modèles non chargés")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
#  Zone de saisie SMILES
# ════════════════════════════════════════════════════════════════════════════

# Initialisation session_state AVANT les widgets
if "smiles_input" not in st.session_state:
    st.session_state["smiles_input"] = ""

examples = {
    "Éthanol"        : "CCO",
    "Benzène"        : "c1ccccc1",
    "Acide acétique" : "CC(=O)O",
    "Octane"         : "CCCCCCCC",
    "Acétone"        : "CC(C)=O",
    "Toluène"        : "Cc1ccccc1",
    "Naphthalène"    : "c1ccc2ccccc2c1",
}

# Boutons traités AVANT le text_area pour que session_state soit à jour au rendu
col_input, col_examples = st.columns([3, 1])

with col_examples:
    st.subheader("Exemples")
    for name, smi in examples.items():
        if st.button(f"➕ {name}", use_container_width=True, key=f"btn_{name}"):
            current = st.session_state["smiles_input"]
            st.session_state["smiles_input"] = (current + "\n" + smi).strip()
            st.rerun()

with col_input:
    st.subheader("🔬  Entrée SMILES")
    # key= synchronise automatiquement le widget avec session_state
    st.text_area(
        "Entrez un ou plusieurs SMILES (un par ligne)",
        height=160,
        placeholder="CCO\nc1ccccc1\nCC(=O)O\nCCCCCCCC",
        help="Notation SMILES standard. Chaque ligne = une molécule.",
        key="smiles_input",
    )

raw_input = st.session_state["smiles_input"]

predict_btn = st.button("▶  Prédire", type="primary", use_container_width=False)

# ════════════════════════════════════════════════════════════════════════════
#  Prédiction
# ════════════════════════════════════════════════════════════════════════════

if predict_btn and raw_input.strip():
    if not models_loaded:
        st.warning("Veuillez d'abord charger les modèles (bouton dans la barre latérale).")
    else:
        pipes    = st.session_state["pipelines"]
        smi_list = [s.strip() for s in raw_input.strip().splitlines() if s.strip()]

        with st.spinner(f"Calcul en cours pour {len(smi_list)} molécule(s)…"):
            df_H, errs_H = predict_smiles(smi_list, *pipes["H"], conv=ADIM_TO_KJMOL)
            df_S, errs_S = predict_smiles(smi_list, *pipes["S"], conv=R_GAS_CONSTANT)

        # ── Erreurs ──────────────────────────────────────────────────────────
        all_errs = list(set(errs_H + errs_S))
        if all_errs:
            st.markdown(
                f'<div class="warn-box">⚠️ SMILES invalides ou descripteurs non-finis '
                f'({len(all_errs)}) : <code>{", ".join(all_errs)}</code></div>',
                unsafe_allow_html=True,
            )

        # ── Fusion des résultats ──────────────────────────────────────────────
        if df_H.empty and df_S.empty:
            st.error("Aucun résultat obtenu. Vérifiez vos SMILES.")
        else:
            if not df_H.empty and not df_S.empty:
                df_merged = df_H.rename(columns={"Valeur prédite": "ΔH°f (kJ/mol)"}).merge(
                    df_S[["SMILES canonique", "Valeur prédite"]].rename(
                        columns={"Valeur prédite": "S° (J/mol·K)"}),
                    on="SMILES canonique", how="outer",
                )
            elif not df_H.empty:
                df_merged = df_H.rename(columns={"Valeur prédite": "ΔH°f (kJ/mol)"})
            else:
                df_merged = df_S.rename(columns={"Valeur prédite": "S° (J/mol·K)"})

            st.markdown("---")
            st.subheader("📊  Résultats")

            # ── Cartes métriques (si une seule molécule) ────────────────────
            if len(df_merged) == 1:
                c1, c2, c3 = st.columns(3)
                smi_can = df_merged["SMILES canonique"].iloc[0]
                img     = mol_to_image(smi_can)
                with c1:
                    if img:
                        st.image(img, caption=smi_can, use_container_width=True)
                with c2:
                    if "ΔH°f (kJ/mol)" in df_merged.columns:
                        val = df_merged["ΔH°f (kJ/mol)"].iloc[0]
                        st.markdown(
                            f'<div class="metric-card"><h4>ΔH°f</h4>'
                            f'<p>{val:.2f} kJ/mol</p></div>',
                            unsafe_allow_html=True)
                with c3:
                    if "S° (J/mol·K)" in df_merged.columns:
                        val = df_merged["S° (J/mol·K)"].iloc[0]
                        st.markdown(
                            f'<div class="metric-card"><h4>S°</h4>'
                            f'<p>{val:.2f} J/mol·K</p></div>',
                            unsafe_allow_html=True)

            # ── Tableau complet ──────────────────────────────────────────────
            st.dataframe(
                df_merged.style.format({
                    "ΔH°f (kJ/mol)": "{:.2f}",
                    "S° (J/mol·K)": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # ── Structures 2D (grille) si plusieurs molécules ───────────────
            if RDKIT_OK and len(df_merged) > 1:
                with st.expander("🖼️  Structures 2D", expanded=False):
                    cols_per_row = 4
                    smiles_col   = df_merged["SMILES canonique"].tolist()
                    rows = [smiles_col[i:i+cols_per_row]
                            for i in range(0, len(smiles_col), cols_per_row)]
                    for row in rows:
                        cols = st.columns(len(row))
                        for col, smi in zip(cols, row):
                            img = mol_to_image(smi)
                            if img:
                                col.image(img, caption=smi[:30], use_container_width=True)

            # ── Téléchargement ───────────────────────────────────────────────
            buf = io.BytesIO()
            df_merged.to_excel(buf, index=False, engine="openpyxl")
            st.download_button(
                label="⬇️  Télécharger Excel",
                data=buf.getvalue(),
                file_name="predictions_thermo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

elif predict_btn and not raw_input.strip():
    st.warning("Veuillez entrer au moins un SMILES.")

# ════════════════════════════════════════════════════════════════════════════
#  Footer
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "Modèle : VAE (encodeur latent) + RDKit descripteurs + ANN régression  •  "
    "T = 298.15 K  •  R = 8.314 J/mol·K"
)
