import json
import time
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from src.model import DenseNet
from src.ood import ood_reason

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(MODELS / "tfidf.joblib")
    calibrator = joblib.load(MODELS / "calibrador.joblib")
    weights = np.load(MODELS / "densa.npz")
    model = DenseNet(hidden=int(weights["hidden"]), seed=int(weights["seed"]))
    model.W1 = weights["W1"]
    model.b1 = weights["b1"]
    model.W2 = weights["W2"]
    model.b2 = weights["b2"]
    decision = json.loads((MODELS / "decision.json").read_text(encoding="utf-8"))
    return vectorizer, calibrator, model, decision


def classify(text, vectorizer, calibrator, model, decision):
    started = time.perf_counter()
    reason = ood_reason(text)
    if reason:
        return {
            "rechazada": True,
            "motivo": reason,
            "clase": "No clasifico (fuera de dominio)",
            "probabilidad": None,
            "umbral": decision["umbral"],
            "segundos": time.perf_counter() - started,
        }
    X = vectorizer.transform([text]).toarray()
    raw = model.predict_proba(X)[0]
    calibrated = float(calibrator.transform(np.array([raw]))[0])
    prediction = int(calibrated >= decision["umbral"])
    return {
        "rechazada": False,
        "motivo": "",
        "clase": "Queja" if prediction == 1 else "Consulta",
        "probabilidad": calibrated,
        "probabilidad_sin_calibrar": float(raw),
        "umbral": decision["umbral"],
        "segundos": time.perf_counter() - started,
    }


def main():
    st.set_page_config(page_title="Clasificador queja / consulta", layout="centered")
    st.title("Clasificador de mensajes: queja vs consulta")
    st.caption("Modelo propio entrenado desde cero con TF-IDF y red densa. Sin LLM ni clasificador preentrenado.")

    vectorizer, calibrator, model, decision = load_artifacts()

    with st.expander("Cómo se decide"):
        st.write(f"Clase positiva: **queja**. Regla: si P(queja) ≥ umbral → Queja; si no → Consulta.")
        st.write(f"Umbral fijado en validación: **{decision['umbral']:.2f}** (criterio: FN cuesta 5, FP cuesta 1).")
        st.write("Si la entrada está vacía, en otro idioma, es demasiado corta, no es del dominio o contiene datos personales, la app la rechaza sin clasificar.")

    text = st.text_area(
        "Escribe un mensaje nuevo",
        height=120,
        placeholder="Ej.: El envío sigue sin llegar y nadie me da una respuesta clara.",
        key="texto",
    )

    col1, col2 = st.columns([1, 3])
    classify_clicked = col1.button("Clasificar", type="primary")
    example_clicked = col2.button("Cargar caso de prueba difícil")

    if example_clicked:
        st.session_state["texto"] = "Sobre la entrega: incompleta y sin novedades."
        st.rerun()

    if classify_clicked:
        if not text.strip():
            st.warning("Escribe un texto antes de clasificar.")
        else:
            result = classify(text, vectorizer, calibrator, model, decision)
            if result["rechazada"]:
                st.error("**Fuera de dominio: no clasifico esta entrada.**")
                st.write(f"Motivo: {result['motivo']}.")
                st.info("Comportamiento esperado: rechazar y no asignar clase ni probabilidad.")
            else:
                if result["clase"] == "Queja":
                    st.error(f"**Clase: {result['clase']}**")
                else:
                    st.success(f"**Clase: {result['clase']}**")
                st.metric("Probabilidad calibrada de queja", f"{result['probabilidad']:.3f}")
                st.write(f"Probabilidad sin calibrar: {result['probabilidad_sin_calibrar']:.3f}")
                st.write(f"Umbral aplicado: **{result['umbral']:.2f}**")
                st.caption(f"Tiempo de respuesta: {result['segundos'] * 1000:.1f} ms")
            st.write(f"Umbral en pantalla: **{result['umbral']:.2f}**")


if __name__ == "__main__":
    main()
