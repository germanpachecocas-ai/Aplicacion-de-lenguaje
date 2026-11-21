import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import OOD_EXAMPLES, generate_corpus, stratified_split

SEED = 42
N = 1020
TRAIN_SIZE = 600
VAL_SIZE = 240


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data_dir = ROOT / "data"
    outputs_dir = ROOT / "outputs"
    data_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    df = generate_corpus(n=N, seed=SEED)
    train, val, test = stratified_split(df, seed=SEED, train_size=TRAIN_SIZE, val_size=VAL_SIZE)

    train = train.copy()
    val = val.copy()
    train["split"] = "train"
    val["split"] = "val"
    dev = pd.concat([train, val], ignore_index=True)
    dev_path = data_dir / "dev.csv"
    test_path = data_dir / "test.csv"
    ood_path = data_dir / "ood.csv"
    dev.to_csv(dev_path, index=False, encoding="utf-8")
    test.to_csv(test_path, index=False, encoding="utf-8")
    pd.DataFrame(OOD_EXAMPLES).to_csv(ood_path, index=False, encoding="utf-8")

    manifest = {
        "seed": SEED,
        "total_ejemplos": len(df),
        "train": len(train),
        "validacion": len(val),
        "prueba": len(test),
        "split": "estratificado 600/240/180, semilla 42; prueba aislada en data/test.csv",
        "etiquetado": "68% con etiqueta derivada de la intención de la plantilla; 32% casos limítrofes con superficie compartida y etiqueta asignada por criterio editorial, con ruido irreducible declarado.",
        "who_labeled": "Etiquetas asignadas por el generador (asistente) según intención de plantilla; German revisó una muestra de 60 casos en ~20 minutos y confirmó el criterio. Limitación del montaje declarada.",
        "datos_personales": "Ninguno: sin nombres, correos, teléfonos ni identificadores.",
        "test_sha256": sha256(test_path),
        "dev_sha256": sha256(dev_path),
        "reglas_uso": "data/test.csv solo se lee una vez desde scripts/final_test.py tras fijar modelo, calibración y umbral en validación.",
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "ejemplos_totales": len(df),
        "textos_unicos": int(df["text"].nunique()),
        "clase_queja": int((df["label"] == 1).sum()),
        "clase_consulta": int((df["label"] == 0).sum()),
        "longitud_promedio_palabras": float(df["text"].str.split().str.len().mean()),
        "ejemplos_ambigua_detectables": int(df["text"].str.contains("No es una queja|No quiero quejarme|Siento confusión|Me molesta", regex=True).sum()),
    }
    (outputs_dir / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
