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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import OOD_EXAMPLES
from src.evaluate import (
    FN_COST,
    FP_COST,
    Calibrator,
    best_threshold,
    cost_per_1000,
    ece,
    reliability,
    rule_baseline,
    threshold_sweep,
    top_errors,
)
from src.model import DenseNet
from src.ood import ood_reason

SEED = 42
COSTS = {"falso_positivo": FP_COST, "falso_negativo": FN_COST}


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_confusion(y, pred, path):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Consulta", "Queja"])
    ax.set_yticks([0, 1], ["Consulta", "Queja"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión - validación")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(sweep_raw, sweep_cal, threshold, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["umbral"] for r in sweep_raw], [r["costo_esperado_por_1000"] for r in sweep_raw], label="Sin calibrar")
    ax.plot([r["umbral"] for r in sweep_cal], [r["costo_esperado_por_1000"] for r in sweep_cal], label="Calibrada")
    ax.axvline(threshold, color="red", linestyle="--", label=f"Umbral elegido {threshold:.2f}")
    ax.set_xlabel("Umbral")
    ax.set_ylabel("Costo esperado por 1000 casos")
    ax.set_title("Barrido de umbral en validación")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(detail_raw, detail_cal, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfecta")
    for detail, label in [(detail_raw, "Sin calibrar"), (detail_cal, "Calibrada")]:
        xs = [b["confianza"] for b in detail]
        ys = [b["exactitud"] for b in detail]
        ax.plot(xs, ys, "o-", label=label)
    ax.set_xlabel("Confianza predicha")
    ax.set_ylabel("Exactitud observada")
    ax.set_title("Curva de fiabilidad - validación")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    started = time.time()
    dev = pd.read_csv(ROOT / "data" / "dev.csv", encoding="utf-8")
    train_df = dev[dev["split"] == "train"].reset_index(drop=True)
    val_df = dev[dev["split"] == "val"].reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        strip_accents="unicode", lowercase=True, max_features=5000,
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    y_train = train_df["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)

    majority_value = int(y_train.mean() >= 0.5)
    majority_pred = np.full(len(y_val), majority_value)
    rule_pred = rule_baseline(val_df["text"].tolist())
    baselines = {
        "clase_mayoritaria": {
            "clase": majority_value,
            "exactitud": float(accuracy_score(y_val, majority_pred)),
            **cost_per_1000(y_val, majority_pred),
        },
        "regla_palabras_clave": {
            "exactitud": float(accuracy_score(y_val, rule_pred)),
            **cost_per_1000(y_val, rule_pred),
        },
    }

    model = DenseNet(hidden=32, lr=0.01, epochs=200, batch=32, l2=1e-4, seed=SEED)
    model.fit(X_train.toarray(), y_train, X_val.toarray(), y_val)
    p_raw = model.predict_proba(X_val.toarray())

    calibrator = Calibrator().fit(y_val, p_raw)
    p_cal = calibrator.transform(p_raw)

    ece_raw, detail_raw = ece(y_val, p_raw)
    ece_cal, detail_cal = ece(y_val, p_cal)
    sweep_raw = threshold_sweep(y_val, p_raw)
    sweep_cal = threshold_sweep(y_val, p_cal)
    chosen = best_threshold(sweep_cal)
    threshold = chosen["umbral"]

    pred_cal = (p_cal >= threshold).astype(int)
    pred_raw_at_05 = (p_raw >= 0.5).astype(int)
    final_metrics = cost_per_1000(y_val, pred_cal)

    error_frame = val_df.copy()
    error_frame["proba"] = p_cal
    error_frame["pred"] = pred_cal
    errors = top_errors(error_frame, n=20)

    ood_table = []
    for item in OOD_EXAMPLES:
        reason = ood_reason(item["text"])
        ood_table.append({
            "entrada": item["text"],
            "respuesta_esperada": item["expected"],
            "respuesta_app": f"Rechazo: {reason}" if reason else "Clasificaría (revisar)",
            "rechazada": bool(reason),
            "motivo": reason or "",
        })

    report = {
        "semilla": SEED,
        "costos": COSTS,
        "tamano": {"train": len(train_df), "validacion": len(y_val)},
        "caracteristicas_tfidf": int(X_train.shape[1]),
        "baselines": baselines,
        "modelo": {
            "arquitectura": "TF-IDF (1-2 gramas) -> densa 32 ReLU -> densa 1 sigmoide, Adam, early stopping en validación",
            "epochs_ejecutadas": len(model.history),
            "mejor_val_loss": min(h["val_loss"] for h in model.history),
            "historial": model.history,
        },
        "calibracion": {
            "metodo": calibrator.method,
            "alternativas": {
                "isotónica": {"ece": calibrator.ece_iso, "descartada": calibrator.method != "isotónica"},
                "platt": {"ece": calibrator.ece_platt, "descartada": calibrator.method != "platt"},
            },
            "criterio": "Se prefiere Platt salvo que isotónica mejore el ECE en más de 0.02; isotónica produce probabilidades 0/1 degeneradas que anulan el barrido de umbral.",
            "ece_antes": calibrator.before_ece,
            "ece_despues": calibrator.after_ece,
            "ece_recalculado_antes": ece_raw,
            "ece_recalculado_despues": ece_cal,
            "fiabilidad_antes": detail_raw,
            "fiabilidad_despues": detail_cal,
        },
        "umbral": {
            "elegido": threshold,
            "criterio": "mínimo costo esperado por 1000 casos en validación con probabilidades calibradas; en empate se elige el umbral más alto para reducir falsas alarmas",
            "mejor_sin_calibrar": best_threshold(sweep_raw),
            "barrido_sin_calibrar": sweep_raw,
            "barrido_calibrado": sweep_cal,
        },
        "metricas_validacion": {
            "exactitud_modelo_umbral_0.5_sin_calibrar": float(accuracy_score(y_val, pred_raw_at_05)),
            "exactitud_modelo_final": float(accuracy_score(y_val, pred_cal)),
            **final_metrics,
        },
        "error_analysis": errors,
        "fuera_de_dominio": ood_table,
        "tiempo_ejecucion_segundos": round(time.time() - started, 2),
    }
    save_json(ROOT / "outputs" / "validation_report.json", report)

    figures = ROOT / "outputs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_confusion(y_val, pred_cal, figures / "confusion_validacion.png")
    plot_sweep(sweep_raw, sweep_cal, threshold, figures / "barrido_umbral.png")
    plot_reliability(detail_raw, detail_cal, figures / "fiabilidad.png")

    joblib.dump(vectorizer, ROOT / "models" / "tfidf.joblib")
    joblib.dump(calibrator, ROOT / "models" / "calibrador.joblib")
    np.savez_compressed(ROOT / "models" / "densa.npz", **model.state())
    decision = {
        "umbral": threshold,
        "costos": COSTS,
        "clase_positiva": "queja",
        "calibracion": calibrator.method,
        "modelo": "densa_tfidf_v1",
        "congelado_en": "validación; prueba no utilizada",
        "max_features": int(X_train.shape[1]),
        "semilla": SEED,
    }
    save_json(ROOT / "models" / "decision.json", decision)

    print(json.dumps({
        "baselines": baselines,
        "calibracion": {"metodo": calibrator.method, "ece_antes": ece_raw, "ece_despues": ece_cal},
        "umbral_elegido": threshold,
        "metricas_validacion": report["metricas_validacion"],
        "tiempo_segundos": report["tiempo_ejecucion_segundos"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
