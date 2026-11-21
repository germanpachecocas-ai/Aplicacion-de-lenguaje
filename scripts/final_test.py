import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import cost_per_1000, ece, rule_baseline
from src.model import DenseNet

MODELS = ROOT / "models"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock = OUTPUTS / "test_run.json"
    if lock.exists():
        print("La prueba ya fue ejecutada. No se vuelve a correr: cualquier reajuste posterior invalida el resultado.")
        print(lock.read_text(encoding="utf-8"))
        return

    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    test_path = DATA / "test.csv"
    if sha256(test_path) != manifest["test_sha256"]:
        raise SystemExit("El hash de test.csv no coincide con el manifest. No se ejecuta la prueba.")

    decision = json.loads((MODELS / "decision.json").read_text(encoding="utf-8"))
    vectorizer = joblib.load(MODELS / "tfidf.joblib")
    calibrator = joblib.load(MODELS / "calibrador.joblib")
    weights = np.load(MODELS / "densa.npz")
    model = DenseNet(hidden=int(weights["hidden"]), seed=int(weights["seed"]))
    model.W1 = weights["W1"]
    model.b1 = weights["b1"]
    model.W2 = weights["W2"]
    model.b2 = weights["b2"]

    test = pd.read_csv(test_path, encoding="utf-8")
    y = test["label"].to_numpy(dtype=int)
    X = vectorizer.transform(test["text"]).toarray()
    p_raw = model.predict_proba(X)
    p_cal = calibrator.transform(p_raw)
    threshold = decision["umbral"]
    pred = (p_cal >= threshold).astype(int)

    majority_value = 0
    majority_pred = np.full(len(y), majority_value)
    rule_pred = rule_baseline(test["text"].tolist())

    ece_raw, _ = ece(y, p_raw)
    ece_cal, _ = ece(y, p_cal)
    started = time.time()
    report = {
        "estado": "corrida final única sobre prueba; sin ajustes posteriores",
        "test_sha256": manifest["test_sha256"],
        "n": len(y),
        "decision_congelada": decision,
        "baselines_test": {
            "clase_mayoritaria": {"exactitud": float(accuracy_score(y, majority_pred)), **cost_per_1000(y, majority_pred)},
            "regla_palabras_clave": {"exactitud": float(accuracy_score(y, rule_pred)), **cost_per_1000(y, rule_pred)},
        },
        "modelo_final": {
            "exactitud": float(accuracy_score(y, pred)),
            **cost_per_1000(y, pred),
        },
        "calibracion_test": {
            "metodo": decision["calibracion"],
            "ece_sin_calibrar": ece_raw,
            "ece_calibrada": ece_cal,
            "mejora_ece": ece_raw - ece_cal,
        },
        "tiempo_total_segundos": round(time.time() - started, 3),
    }
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "test_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lock.write_text(json.dumps({
        "ejecutado": True,
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_sha256": manifest["test_sha256"],
        "advertencia": "No volver a ejecutar ni ajustar el modelo después de esta corrida.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    cm = cost_per_1000(y, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Consulta", "Queja"])
    ax.set_yticks([0, 1], ["Consulta", "Queja"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión - prueba final")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    (OUTPUTS / "figures").mkdir(exist_ok=True)
    fig.savefig(OUTPUTS / "figures" / "confusion_prueba.png", dpi=150)
    plt.close(fig)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
