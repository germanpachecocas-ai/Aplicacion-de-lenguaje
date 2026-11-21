import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SLOTS = {
    "producto": ["pedido", "paquete", "suscripción", "tablet", "impresora", "teléfono", "laptop", "cafetera", "bicicleta", "monitor", "taladro", "microondas", "router", "silla", "cámara"],
    "producto_det": ["el pedido", "el paquete", "la suscripción", "la tablet", "la impresora", "el teléfono", "la laptop", "la cafetera", "la bicicleta", "el monitor", "el taladro", "el microondas", "el router", "la silla", "la cámara"],
    "estado": ["dañado", "tardío", "incompleto", "con fallas", "mojado", "abierto", "rayado"],
    "accion": ["quiero una solución", "necesito un reembolso", "esperaba otra cosa", "no acepto esto", "pido una aclaración", "voy a reportarlo", "requiero respuesta", "es inaceptable"],
    "tiempo": ["dos días", "una semana", "desde ayer", "varias horas", "más de una semana", "desde el lunes"],
    "situacion": ["mi pedido", "la entrega", "el soporte", "mi cuenta", "la devolución", "el cobro", "la garantía", "mi solicitud", "el envío", "la factura"],
    "respuesta": ["responde", "da una explicación", "se comunica", "resuelve nada", "contesta"],
    "problema_cobro": ["un cargo duplicado", "un monto incorrecto", "un cobro extra", "una comisión no anunciada", "un descuento que no aplicaron"],
    "adj": ["pésima", "decepcionante", "lenta", "confusa", "incoherente", "insuficiente", "mala"],
    "funcion": ["funciona", "enciende", "conecta", "carga"],
    "momento": ["ayer", "desde el primer día", "hace una semana", "después de la actualización"],
    "defecto": ["un golpe", "rayas", "un error", "piezas sueltas", "un defecto", "un ruido raro"],
    "info": ["precio", "plazo de entrega", "disponibilidad", "garantía", "horario", "características", "manual", "compatibilidad", "política de devolución", "costo de envío", "tiempo de respuesta", "cobertura"],
    "accion_consulta": ["hacer un pedido", "seguir mi envío", "cambiar mi dirección", "solicitar una devolución", "pagar", "activar la garantía", "contactar soporte", "obtener el manual", "cancelar un pedido"],
    "lugar": ["la tienda", "el catálogo", "sucursales", "la web", "la app"],
    "medio": ["transferencia", "tarjeta", "efectivo", "el portal"],
    "servicio": ["soporte", "almacén", "mostrador"],
}

COMPLAINT_TEMPLATES = [
    "{producto_det} llegó {estado} y {accion}.",
    "Llevo {tiempo} esperando {situacion} y nadie {respuesta}.",
    "Me cobraron {problema_cobro} y {accion}.",
    "La atención fue {adj} y {accion}.",
    "{producto_det} no {funcion} desde {momento}.",
    "Solicité la cancelación de {situacion} y todavía {accion}.",
    "La garantía no {funcion} y {accion}.",
    "{producto_det} tiene {defecto} y {accion}.",
    "Me parece {adj} que {situacion}, {accion}.",
    "Nadie {respuesta} mi {situacion} después de {tiempo}.",
    "El servicio {estado} y {accion}.",
    "Recibí {situacion} de forma incorrecta y {accion}.",
    "Hay {problema_cobro} en mi factura y {accion}.",
    "{producto_det} se rompió y tiene {defecto}, {accion}.",
    "Es {adj} que nadie {respuesta} sobre {situacion}.",
]

QUERY_TEMPLATES = [
    "¿Cuál es el detalle de {info} de {producto_det}?",
    "Quisiera saber {info} sobre {situacion}.",
    "¿Cómo puedo {accion_consulta} mi {situacion}?",
    "Tengo una duda sobre {info} de {producto}.",
    "¿Dónde puedo {accion_consulta} {situacion}?",
    "Me gustaría conocer {info} antes de {accion_consulta}.",
    "¿Cuánto cuesta {producto}?",
    "¿Puedo {accion_consulta} en {lugar}?",
    "Necesito información sobre {info}.",
    "Quiero conocer {info} para {accion_consulta}.",
    "Buen día, ¿podrían indicarme la información sobre {info}?",
    "Consulta: ¿se puede {accion_consulta} con {medio}?",
    "Qué {info} tiene {producto}?",
    "Quiero saber si {situacion} incluye {info}.",
    "¿A qué hora puedo {accion_consulta} en {lugar}?",
]

NEUTRAL_COMPLAINT_TEMPLATES = [
    "¿Podrían decirme por qué {situacion} no {funcion}?",
    "¿Cuánto tiempo más tengo que esperar {situacion}?",
    "¿Es normal que {situacion} esté {estado}?",
    "¿No deberían resolver {situacion} sin costo?",
    "Me gustaría entender por qué {situacion} {estado}.",
    "Seguimos sin {situacion} y quisiera una respuesta.",
]

NEUTRAL_QUERY_TEMPLATES = [
    "Si {situacion} está {estado}, ¿cómo {accion_consulta}?",
    "¿Qué hago si {situacion} no {funcion}?",
    "Quiero saber el costo si hay {problema_cobro}.",
    "Me preocupa {situacion}, ¿cuál es el detalle de {info}?",
    "¿Hay que {accion_consulta} aunque {situacion} esté {estado}?",
    "En caso de {situacion} {estado}, ¿cuál es el detalle de {info}?",
]

AMBIGUOUS_COMPLAINT_TEMPLATES = [
    "No es una queja, pero el {producto} llegó {estado} y nadie {respuesta}.",
    "Solo pregunto: ¿por qué {situacion} sigue {estado}?",
    "Consulta urgente: {situacion} {estado} y necesito una solución.",
    "No quiero quejarme, aunque {situacion} {estado} desde {momento}.",
]

AMBIGUOUS_QUERY_TEMPLATES = [
    "Me molesta no saber {info}, ¿podrían decirme?",
    "Tengo un problema con {situacion}, ¿cómo {accion_consulta}?",
    "No entiendo {info}: ¿me explican?",
    "Siento confusión con {situacion}, ¿cuál es el {info}?",
]

SHARED_TEMPLATES = [
    "{situacion} {estado} y no tengo respuesta.",
    "{situacion} {estado}, ¿qué sigue?",
    "Sobre {situacion}: {estado} y sin novedades.",
    "{producto_det} {estado} desde {momento}.",
    "Me comentan que {situacion} {estado}.",
]

TYPOS = {
    "producto": ["prodcuto", "produto"],
    "llegó": ["llego", "llegoo"],
    "tardío": ["tardio"],
    "suscripción": ["suscripcion"],
    "teléfono": ["telefono"],
    "política": ["politica"],
    "necesito": ["necesito", "nesesito"],
    "respuesta": ["respuesta", "respueta"],
    "cobro": ["cobro", "cobroo"],
    "información": ["informacion"],
}

OOD_EXAMPLES = [
    {"text": "", "expected": "Rechazar: entrada vacía; no clasificar."},
    {"text": "   ", "expected": "Rechazar: solo espacios; no clasificar."},
    {"text": "hola", "expected": "Rechazar: texto demasiado corto y sin contexto de dominio."},
    {"text": "The item arrived damaged and I demand a refund", "expected": "Rechazar: otro idioma; no clasificar."},
    {"text": "¿Cómo está el clima en Marte?", "expected": "Rechazar: fuera de dominio de atención a cliente."},
    {"text": "asdf qwerty zxcv", "expected": "Rechazar: texto sin contenido lingüístico útil."},
    {"text": "😐😑🙄", "expected": "Rechazar: sin texto alfabético."},
    {"text": "Noticias del partido de ayer y el campeonato de tenis", "expected": "Rechazar: tema ajeno al dominio."},
    {"text": "Consulta sobre el precio del pan en la esquina", "expected": "Rechazar: dominio distinto (no es atención a cliente de producto)."},
    {"text": "El señor García llamó desde el número 555-1234", "expected": "Rechazar: contiene datos personales; no clasificar y no almacenar."},
]


def _apply_noise(text, rng):
    words = text.split()
    if not words:
        return text
    for i, word in enumerate(words):
        clean = word.strip("¿?.,;:!")
        if clean in TYPOS and rng.random() < 0.16:
            replacement = rng.choice(TYPOS[clean])
            words[i] = word.replace(clean, replacement)
    result = " ".join(words)
    if rng.random() < 0.08:
        result = result.replace(".", "..")
    if rng.random() < 0.12:
        result = result.lower()
    return result


def _fill(template, rng):
    result = template
    for key in SLOTS:
        while "{" + key + "}" in result:
            result = result.replace("{" + key + "}", rng.choice(SLOTS[key]), 1)
    return result


def generate_corpus(n=900, seed=42):
    if n < 500:
        raise ValueError("Se requieren al menos 500 ejemplos")
    rng = np.random.default_rng(seed)
    labels = np.array([1, 0] * (n // 2) + ([1] if n % 2 else []))[:n]
    rng.shuffle(labels)
    rows = []
    for i, label in enumerate(labels):
        draw = rng.random()
        if draw < 0.32:
            label = int(rng.integers(0, 2))
            template = rng.choice(SHARED_TEMPLATES)
        elif draw < 0.45:
            template = rng.choice(AMBIGUOUS_COMPLAINT_TEMPLATES if label == 1 else AMBIGUOUS_QUERY_TEMPLATES)
        elif draw < 0.70:
            template = rng.choice(NEUTRAL_COMPLAINT_TEMPLATES if label == 1 else NEUTRAL_QUERY_TEMPLATES)
        else:
            template = rng.choice(COMPLAINT_TEMPLATES if label == 1 else QUERY_TEMPLATES)
        text = _apply_noise(_fill(template, rng), rng)
        rows.append({"id": f"ex-{i:04d}", "text": text, "label": int(label), "origen": "sintetico_revisado"})
    return pd.DataFrame(rows)


def stratified_split(df, seed=42, train_size=600, val_size=120):
    test_size = len(df) - train_size - val_size
    if test_size < 100:
        raise ValueError("El conjunto de prueba debe tener al menos 100 ejemplos")
    rest, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    train, val = train_test_split(
        rest, train_size=train_size, random_state=seed, stratify=rest["label"]
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
