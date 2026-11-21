import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

FP_COST = 1
FN_COST = 5


def cost_per_1000(y_true, y_pred, fp_cost=FP_COST, fn_cost=FN_COST):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "costo_esperado_por_1000": float((fp * fp_cost + fn * fn_cost) / len(y_true) * 1000),
        "costo_total": int(fp * fp_cost + fn * fn_cost),
    }


def threshold_sweep(y_true, proba, fp_cost=FP_COST, fn_cost=FN_COST):
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    rows = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = (proba >= threshold).astype(int)
        result = cost_per_1000(y_true, pred, fp_cost, fn_cost)
        rows.append({"umbral": float(threshold), **result})
    return rows


def best_threshold(sweep):
    minimum = min(row["costo_esperado_por_1000"] for row in sweep)
    tied = [row for row in sweep if abs(row["costo_esperado_por_1000"] - minimum) < 1e-9]
    return max(tied, key=lambda row: row["umbral"])


def ece(y_true, proba, bins=10):
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    confidence = np.maximum(proba, 1 - proba)
    prediction = (proba >= 0.5).astype(int)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    detail = []
    for i in range(bins):
        mask = (confidence >= edges[i]) & (confidence < edges[i + 1] if i < bins - 1 else confidence <= edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float((y_true[mask] == prediction[mask]).mean())
        conf = float(confidence[mask].mean())
        total += count / len(y_true) * abs(acc - conf)
        detail.append({"bin": i, "n": count, "confianza": conf, "exactitud": acc, "gap": acc - conf})
    return float(total), detail


def reliability(y_true, proba, bins=10):
    _, detail = ece(y_true, proba, bins)
    return detail


class Calibrator:
    def __init__(self):
        self.method = None
        self.isotonic = None
        self.platt = None
        self.before_ece = None
        self.after_ece = None
        self.ece_iso = None
        self.ece_platt = None

    def fit(self, y_true, proba):
        y = np.asarray(y_true, dtype=int)
        p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
        self.before_ece, _ = ece(y, p)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p, y)
        p_iso = np.clip(iso.predict(p), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        platt.fit(logit, y)
        p_platt = platt.predict_proba(logit)[:, 1]
        self.ece_iso, _ = ece(y, p_iso)
        self.ece_platt, _ = ece(y, p_platt)
        if self.ece_iso < self.ece_platt - 0.02:
            self.method = "isotónica"
            self.isotonic = iso
            self.after_ece = self.ece_iso
        else:
            self.method = "platt"
            self.platt = platt
            self.after_ece = self.ece_platt
        return self

    def transform(self, proba):
        if self.method is None:
            return np.asarray(proba, dtype=float)
        p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "isotónica":
            return np.clip(self.isotonic.predict(p), 0.0, 1.0)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        return self.platt.predict_proba(logit)[:, 1]


KEYWORDS_COMPLAINT = {"queja", "dañado", "daniado", "fallas", "falla", "tarde", "tardío", "tardio", "reembolso", "pésima", "pesima", "mala", "no", "nunca", "problema"}


def rule_baseline(texts):
    return np.array([
        1 if KEYWORDS_COMPLAINT & set(str(t).lower().replace("¿", "").replace("?", "").replace(".", "").split()) else 0
        for t in texts
    ], dtype=int)


def categorize_error(text):
    lower = str(text).lower()
    if any(k in lower for k in ["sin novedades", "no tengo respuesta", "qué sigue", "me comentan que"]):
        return "etiqueta ambigua con superficie compartida", "La misma forma léxica aparece con ambas etiquetas en los datos; existe ruido irreducible que ningún modelo puede resolver."
    if any(k in lower for k in ["no es una queja", "no quiero quejarme", "siento confusión", "me molesta", "solo pregunto", "consulta urgente"]):
        return "etiqueta ambigua", "El texto mezcla intención de consulta y contenido de queja; el límite entre clases no está bien definido para ese caso."
    if any(k in lower for k in ["llego", "respueta", "prodcuto", "politica", "telefono", "suscripcion", ".."]):
        return "texto con faltas de ortografía", "La falta ortográfica debilita las señales léxicas del TF-IDF y reduce la probabilidad de la clase correcta."
    if len(str(text).split()) < 6:
        return "texto demasiado corto", "Hay pocas señales léxicas para discriminar la clase."
    if "consulta" in lower and "no" in lower:
        return "negación o doble sentido", "La negación invierte la lectura de una palabra de queja dentro de una consulta."
    if ("precio" in lower or "garantía" in lower or "cómo" in lower) and any(k in lower for k in ["dañado", "fallas", "cobro", "tarde"]):
        return "clase confundida con otra", "El texto contiene indicios fuertes de ambas clases y el modelo pondera más las señales de consulta."
    return "otro", "No se identificó una causa única; requiere revisión manual del caso."


def top_errors(frame, text_col="text", label_col="label", prob_col="proba", pred_col="pred", n=20):
    wrong = frame[frame[label_col] != frame[pred_col]].copy()
    wrong["confianza_error"] = np.where(
        wrong[label_col] == 1, 1 - wrong[prob_col], wrong[prob_col]
    )
    wrong = wrong.sort_values("confianza_error", ascending=False).head(n)
    rows = []
    for _, row in wrong.iterrows():
        category, hypothesis = categorize_error(row[text_col])
        rows.append({
            "id": row.get("id", ""),
            "texto": row[text_col],
            "clase_real": int(row[label_col]),
            "clase_predicha": int(row[pred_col]),
            "probabilidad": float(row[prob_col]),
            "tipo_falla": category,
            "hipotesis": hypothesis,
        })
    return rows
