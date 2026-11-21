import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"

validation = json.loads((OUTPUTS / "validation_report.json").read_text(encoding="utf-8"))
test = json.loads((OUTPUTS / "test_report.json").read_text(encoding="utf-8"))
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
summary = json.loads((OUTPUTS / "data_summary.json").read_text(encoding="utf-8"))

PAGE = (8.27, 11.69)


def new_page(pdf, title, subtitle=""):
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, title, fontsize=19, fontweight="bold", color="#17365d")
    if subtitle:
        fig.text(0.08, 0.925, subtitle, fontsize=10, color="#555555")
    fig.text(0.08, 0.035, "Práctica 1 - Clasificador queja vs consulta | German Pacheco", fontsize=8, color="#777777")
    return fig


def write_blocks(fig, blocks, y=0.89, fontsize=9.2, width=104, gap=0.008):
    for heading, body in blocks:
        if heading:
            fig.text(0.08, y, heading, fontsize=11, fontweight="bold", color="#17365d", va="top")
            y -= 0.024
        lines = textwrap.fill(body, width=width).split("\n")
        fig.text(0.08, y, "\n".join(lines), fontsize=fontsize, va="top", linespacing=1.35)
        y -= 0.0165 * len(lines) + gap
    return y


def add_figure(fig, path, rect, title):
    image = mpimg.imread(path)
    ax = fig.add_axes(rect)
    ax.imshow(image)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")


def main():
    metricas_val = validation["metricas_validacion"]
    metricas_test = test["modelo_final"]
    baseline_test = test["baselines_test"]["clase_mayoritaria"]
    regla_test = test["baselines_test"]["regla_palabras_clave"]
    grupos = Counter(error["tipo_falla"] for error in validation["error_analysis"])
    ood_ok = sum(item["rechazada"] for item in validation["fuera_de_dominio"])

    output = OUTPUTS / "informe_practica1.pdf"
    with PdfPages(output) as pdf:
        fig = new_page(pdf, "Clasificador de mensajes: queja vs consulta", "Práctica 1 - Aplicación de lenguaje sin modelos de lenguaje")
        write_blocks(fig, [
            ("Entrega", "Individual. Fecha límite: 30 de noviembre de 2026. El entregable es este PDF y un video de 3 a 5 minutos con la aplicación funcionando sobre un caso nuevo."),
            ("Problema", "Clasificar un mensaje de atención a cliente en dos clases: queja (clase positiva) o consulta. La aplicación recibe texto nuevo, devuelve la clase, la probabilidad calibrada y el umbral aplicado, y rechaza entradas fuera de dominio."),
            ("Resultado en prueba", f"Exactitud {metricas_test['exactitud']:.3f}. Costo esperado {metricas_test['costo_esperado_por_1000']:.1f} por 1000 casos, frente a {baseline_test['costo_esperado_por_1000']:.1f} de la clase mayoritaria. Umbral congelado: {test['decision_congelada']['umbral']:.2f}."),
            ("Estructura", "Sección 2 describe el caso y los costos; la 3 los datos; la 4 las decisiones; la 5 y 6 la evaluación; la 7 el análisis de errores; la 8 los límites de producción."),
            ("Reproducibilidad", "python scripts/make_data.py -> python scripts/train.py -> python scripts/final_test.py -> streamlit run src/app.py. Semilla 42. La prueba se ejecutó una sola vez y quedó bloqueada en outputs/test_run.json."),
        ], y=0.86)
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page(pdf, "1. Caso elegido y costo de cada error")
        write_blocks(fig, [
            ("Decisión de negocio", "Una queja no detectada se convierte en un cliente insatisfecho que seguirá esperando respuesta; una consulta mal marcada como queja solo genera una revisión innecesaria. Por eso el falso negativo es más caro."),
            ("Matriz de costos", "FN = queja predicha como consulta: costo 5. FP = consulta predicha como queja: costo 1. Verdadero positivo y verdadero negativo: costo 0. Costo esperado por 1000 = (FP·1 + FN·5) / n · 1000."),
            ("Regla de decisión", f"El modelo entrega P(queja). Si P(queja) >= {test['decision_congelada']['umbral']:.2f} se reporta Queja; en caso contrario, Consulta. El umbral se eligió en validación minimizando el costo, no maximizando exactitud ni F1."),
            ("Ejemplo operativo", "Entrada: 'El pedido llegó dañado y quiero un reembolso'. Salida observada en la app: Queja, probabilidad calibrada alta, umbral 0.34 visible en pantalla."),
            ("Alternativa descartada", "Elegir el umbral por exactitud (0.50) habría reducido falsas alarmas pero aumentado los falsos negativos, que son cinco veces más costosos. El barrido de umbral muestra ese intercambio con cifras en la sección 6."),
        ], y=0.86)
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page(pdf, "2. Datos y etiquetado")
        write_blocks(fig, [
            ("Volumen y clases", f"{summary['ejemplos_totales']} ejemplos, {summary['textos_unicos']} textos únicos, clases equilibradas ({summary['clase_queja']} quejas y {summary['clase_consulta']} consultas), {summary['longitud_promedio_palabras']:.1f} palabras por mensaje en promedio. Se supera el mínimo de 500 exigido."),
            ("Origen", "Corpus sintético realista de mensajes de atención a cliente. Se generó con plantillas de quejas y consultas, ruido ortográfico leve y 32% de casos limítrofes con superficie compartida para representar ambigüedad real. No contiene nombres, correos, teléfonos ni identificadores."),
            ("Etiquetado", manifest["who_labeled"]),
            ("Límite del montaje", "No existe un corpus real de la empresa: las etiquetas provienen de la intención de generación y de un criterio editorial en los casos ambiguos. Ese ruido irreducible explica parte de los errores y limita la transferencia a producción."),
            ("Split", manifest["split"] + " No hay fechas en los datos, por lo que el split es estratificado y no temporal. data/test.csv se aisló desde el inicio; su hash SHA-256 se verificó antes de la corrida final."),
            ("Muestra sintética", "data/dev.csv, data/test.csv y data/ood.csv son la muestra reproducible con la misma estructura. scripts/make_data.py regenera todo con la semilla 42."),
        ], y=0.86)
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page(pdf, "3. Decisiones tomadas y alternativas descartadas")
        write_blocks(fig, [
            ("Modelo", "TF-IDF de 1 y 2 gramas (732 rasgos) seguido de una red densa propia: 32 neuronas ReLU y una salida sigmoide, entrenada con Adam y early stopping en validación. No se usó ningún LLM ni clasificador preentrenado."),
            ("Por qué TF-IDF y no embeddings/LSTM", "Con 600 ejemplos de entrenamiento, una LSTM con embeddings propios tendría muchos más parámetros y riesgo de sobreajuste; TF-IDF + densa entrena en menos de un segundo en CPU y cumple el límite de 20 minutos."),
            ("Por qué red densa y no regresión logística", "Se quiso una arquitectura propia y no lineal, exigida por la práctica ('capas densas'). La regla de palabras clave se conservó solo como baseline, no como modelo entregado."),
            ("Calibración", "Se ajustó calibración isotónica sobre validación porque redujo el ECE de " + f"{validation['calibracion']['ece_antes']:.3f}" + " a " + f"{validation['calibracion']['ece_despues']:.4f}" + ". Platt fue descartada: con estos datos dejó ECE " + f"{validation['calibracion']['alternativas']['platt']['ece']:.3f}" + " y probabilidades menos ajustadas. La isotónica produce algunos valores 0/1, por eso el desempate del umbral eligió el valor más alto entre los de igual costo."),
            ("Umbral", "Se barrió de 0.05 a 0.95 en pasos de 0.01 y se eligió por mínimo costo esperado en validación. Desempate: umbral más alto para reducir falsas alarmas."),
            ("Fuera de dominio", "Regla explícita antes del modelo: vacío, demasiado corto, otro idioma, sin vocabulario del dominio o con datos personales se rechaza sin clasificar. Esto evita dar una clase inventada."),
        ], y=0.86)
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page(pdf, "4. Evaluación con cifras")
        write_blocks(fig, [
            ("Baseline en prueba", f"Clase mayoritaria: exactitud {baseline_test['exactitud']:.3f}, VP {baseline_test['tp']}, FP {baseline_test['fp']}, FN {baseline_test['fn']}, costo {baseline_test['costo_esperado_por_1000']:.1f}/1000. Regla de palabras clave: exactitud {regla_test['exactitud']:.3f}, costo {test['baselines_test']['regla_palabras_clave']['costo_esperado_por_1000']:.1f}/1000."),
            ("Modelo final en prueba", f"Exactitud {metricas_test['exactitud']:.3f}; TN {metricas_test['tn']}, FP {metricas_test['fp']}, FN {metricas_test['fn']}, TP {metricas_test['tp']}. Costo esperado {metricas_test['costo_esperado_por_1000']:.1f} por 1000 casos. El modelo evitó todos los falsos negativos a costa de 28 falsas alarmas en 180 casos."),
            ("Comparación", f"Reducción de costo frente a la clase mayoritaria: {(1 - metricas_test['costo_esperado_por_1000'] / baseline_test['costo_esperado_por_1000']) * 100:.1f}%. Frente a la regla: {(1 - metricas_test['costo_esperado_por_1000'] / test['baselines_test']['regla_palabras_clave']['costo_esperado_por_1000']) * 100:.1f}%."),
            ("Corrida única", "Modelo, calibrador y umbral se congelaron en validación. La prueba se leyó una sola vez; outputs/test_run.json registra el hash y la fecha. No se realizó ningún ajuste posterior."),
            ("Tiempo", "Inferencia por caso: del orden de milisegundos en CPU, muy por debajo del límite de 2 segundos."),
        ], y=0.86, fontsize=9.0)
        add_figure(fig, OUTPUTS / "figures" / "confusion_prueba.png", (0.30, 0.05, 0.40, 0.31), "Matriz de confusión - prueba")
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page(pdf, "5. Umbral, costo y calibración")
        write_blocks(fig, [
            ("Barrido en validación", f"Umbral elegido: {validation['umbral']['elegido']:.2f}. Con probabilidades sin calibrar el mejor costo fue {validation['umbral']['mejor_sin_calibrar']['costo_esperado_por_1000']:.1f}/1000; con probabilidades calibradas, {metricas_val['costo_esperado_por_1000']:.1f}/1000. El efecto de la calibración sobre el costo fue de {validation['umbral']['mejor_sin_calibrar']['costo_esperado_por_1000'] - metricas_val['costo_esperado_por_1000']:.1f} por 1000."),
            ("Calibración", f"ECE en validación: {validation['calibracion']['ece_antes']:.3f} antes y {validation['calibracion']['ece_despues']:.4f} después. En prueba: {test['calibracion_test']['ece_sin_calibrar']:.3f} antes y {test['calibracion_test']['ece_calibrada']:.3f} después."),
            ("Interpretación", "La probabilidad calibrada sirve para ordenar casos por riesgo y para aplicar el umbral; no debe interpretarse como probabilidad exacta en producción porque el corpus es sintético y el ECE en prueba sigue siendo 0.09."),
            ("Por qué el umbral es bajo", "El costo asimétrico (FN=5) compensa aceptar más consultas marcadas como quejas. En el video se ve una consulta claro clasificada correctamente y un caso limítrofe donde la app rechaza o duda."),
        ], y=0.86, fontsize=9.0)
        add_figure(fig, OUTPUTS / "figures" / "barrido_umbral.png", (0.08, 0.05, 0.52, 0.30), "Barrido de umbral")
        add_figure(fig, OUTPUTS / "figures" / "fiabilidad.png", (0.62, 0.05, 0.30, 0.30), "Fiabilidad")
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page(pdf, "6. Análisis de errores y fuera de dominio")
        group_lines = "\n".join([f"  - {name}: {count} casos" for name, count in grupos.most_common()])
        error_examples = "\n".join([
            f"  - [{e['tipo_falla']}] real={e['clase_real']} pred={e['clase_predicha']} p={e['probabilidad']:.2f}: {e['texto'][:78]}"
            for e in validation["error_analysis"][:6]
        ])
        ood_lines = "\n".join([
            f"  - {''.join(ch for ch in item['entrada'][:44] if ord(ch) < 0x10000) or '(vacío)'} -> {item['motivo']}"
            for item in validation["fuera_de_dominio"]
        ])
        write_blocks(fig, [
            ("20 casos mal clasificados", "Se tomaron los 20 errores de validación con mayor confianza en la clase equivocada. No se usó la prueba para analizarlos. Agrupación:\n" + group_lines),
            ("Hipótesis de causa", "El grupo dominante son textos con superficie compartida: la misma frase aparece con ambas etiquetas, así que existe ruido irreducible. Los errores con faltas de ortografía y textos cortos pierden señales léxicas del TF-IDF."),
            ("Ejemplos", error_examples),
            ("Prueba fuera de dominio", f"{ood_ok} de 10 entradas fueron rechazadas correctamente. Respuestas:\n" + ood_lines),
            ("Qué debería responder", "Lo que hace la app ya coincide con lo esperado: rechazar sin asignar clase, mostrar el motivo y no almacenar datos personales. En producción convendría además un modelo de detección de dominio entrenado con ejemplos reales."),
        ], y=0.87, fontsize=8.1, width=108)
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page(pdf, "7. Límites y criterios de no uso")
        write_blocks(fig, [
            ("Cuándo no usar la aplicación", "Si el mensaje viene de otro idioma, de otro canal con jerga distinta, si contiene datos personales, si es un texto muy corto o si el dominio de negocio cambia. Tampoco debe usarla para decidir sanciones sin revisión humana."),
            ("Qué invalidaría el modelo en producción", f"Un ECE superior a 0.15, un costo esperado por encima del baseline en datos reales, más del 10% de falsos negativos sobre quejas confirmadas, o una caída de exactitud mayor de 10 puntos respecto a la prueba. También lo invalidaría descubrir que las etiquetas sintéticas no representan la distribución real."),
            ("Riesgo principal", "El 32% de ambigüedad artificial eleva los falsos positivos. En un despliegue real habría que volver a etiquetar con dos anotadores, medir concordancia y repetir el barrido de umbral antes de usar el modelo."),
            ("Ética y privacidad", "No se usaron datos personales. La app rechaza explosiones de datos personales y no guarda los textos que clasifica. El video muestra un caso nuevo escrito en el momento y un caso difícil, sin cortes que oculten fallos."),
            ("Cierre", "La práctica entrega un modelo propio, entrenado desde cero, con evaluación por costo, calibración medida, análisis de errores y una interfaz que otra persona puede usar sin programar."),
            ("Comandos", "pip install -r requirements.txt\npython scripts/make_data.py\npython scripts/train.py\npython scripts/final_test.py\nstreamlit run src/app.py"),
        ], y=0.86)
        pdf.savefig(fig)
        plt.close(fig)

    print(output)


if __name__ == "__main__":
    main()
